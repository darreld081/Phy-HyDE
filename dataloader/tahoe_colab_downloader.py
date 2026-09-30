#!/usr/bin/env python3
"""Prepare, randomly select, and export a Tahoe subset using remote range reads.

Requires: pyarrow, huggingface_hub. No expression shard is cached on disk.
Selection uses the small obs table first; export retrieves matching expression
row groups. Network transfer can be much larger than the saved subset.
"""
import argparse
import ast
import bisect
from collections import Counter, defaultdict
import hashlib
import heapq
import io
import json
from pathlib import Path
import random
import shutil
import urllib.request
import zipfile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO = "tahoebio/Tahoe-100M"
CONTROL = "DMSO_TF"
META = ["sample_metadata", "drug_metadata", "gene_metadata", "cell_line_metadata"]


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def open_remote(fs, revision, path):
    # Only an in-memory read-ahead cache; never simplecache/filecache.
    return fs.open(f"datasets/{REPO}@{revision}/{path}", "rb",
                   block_size=1024 * 1024, cache_type="readahead")


def round_robin(groups, n, seed):
    rng = random.Random(seed)
    groups = {k: sorted(set(v)) for k, v in groups.items()}
    keys = sorted(groups)
    rng.shuffle(keys)
    for values in groups.values():
        rng.shuffle(values)
    result = []
    while len(result) < n:
        changed = False
        for key in keys:
            if groups[key] and len(result) < n:
                result.append(groups[key].pop())
                changed = True
        if not changed:
            raise ValueError(f"Only {len(result)} distinct items available")
    return result


def prepare(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise SystemExit("plan.json exists: use another --out directory to prepare a new plan.")
    api = HfApi()
    revision = api.dataset_info(REPO).sha
    meta = root / "metadata"
    meta.mkdir(exist_ok=True)
    tables = {}
    for name in META:
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/metadata/{name}.parquet"
        with urllib.request.urlopen(url, timeout=120) as response:
            (meta / f"{name}.parquet").write_bytes(response.read())
        tables[name] = pq.read_table(meta / f"{name}.parquet").to_pylist()
    files = sorted(x.path for x in api.list_repo_tree(REPO, repo_type="dataset",
                   revision=revision, path_in_repo="data", recursive=True)
                   if x.path.endswith(".parquet"))
    fs = HfFileSystem()
    # The first shard contains all 50 lines in the verified release. This check
    # fails rather than silently selecting from the metadata's 102-line superset.
    with open_remote(fs, revision, files[0]) as source:
        actual = set(pc.unique(pq.ParquetFile(source).read(columns=["cell_line_id"])
                               ["cell_line_id"]).to_pylist())
    if len(actual) != 50:
        raise SystemExit("First shard no longer contains 50 lines; inspect the release before preparing.")
    line_groups = defaultdict(list)
    for row in tables["cell_line_metadata"]:
        if row["Cell_ID_Cellosaur"] in actual:
            line_groups[row["Organ"] or "unknown"].append(row["Cell_ID_Cellosaur"])
    lines = round_robin(line_groups, args.lines, args.seed)
    drug_groups = defaultdict(list)
    available = {r["drug"] for r in tables["sample_metadata"] if r["drug"] != CONTROL}
    for row in tables["drug_metadata"]:
        if row["drug"] in available:
            drug_groups[row["moa-fine"] or "unclear"].append(row["drug"])
    drugs = round_robin(drug_groups, args.drugs, args.seed)
    plan = dict(repo=REPO, revision=revision, seed=args.seed, cell_line_ids=lines,
                drugs=drugs, cells_per_sample=args.cells, control_cells_per_sample=args.controls,
                max_output_gb=args.max_gb, min_genes=args.min_genes,
                max_mito_fraction=args.max_mito, require_pass_filter="full")
    save_json(root / "plan.json", plan)
    save_json(root / "expression_files.json", files)
    describe(root)


def describe(root):
    plan = json.loads((root / "plan.json").read_text())
    samples = pq.read_table(root / "metadata/sample_metadata.parquet").to_pylist()
    chosen = [r for r in samples if r["drug"] in plan["drugs"]]
    plates = {r["plate"] for r in chosen}
    controls = [r for r in samples if r["drug"] == CONTROL and r["plate"] in plates]
    total = len(plan["cell_line_ids"]) * (len(chosen) * plan["cells_per_sample"] +
                                         len(controls) * plan["control_cells_per_sample"])
    print(f"{len(plan['cell_line_ids'])} lines; {len(plan['drugs'])} drugs; "
          f"{len(chosen)} treatment samples; {len(plates)} plates; "
          f"{len(controls)} control samples; at most {total:,} cells.", flush=True)
    return plan, chosen + controls


def selection_checkpoint(root, plan, heaps, counts, next_group, total_groups):
    columns = {"sample": [], "cell_line_id": [], "BARCODE_SUB_LIB_ID": [], "priority": []}
    for (sample, line), heap in heaps.items():
        for negative_rank, barcode in heap:
            columns["sample"].append(sample)
            columns["cell_line_id"].append(line)
            columns["BARCODE_SUB_LIB_ID"].append(barcode)
            columns["priority"].append(-negative_rank)
    schema = pa.schema([("sample", pa.string()), ("cell_line_id", pa.string()),
                        ("BARCODE_SUB_LIB_ID", pa.string()), ("priority", pa.uint64())])
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pydict(columns, schema=schema), sink, compression="zstd")
    state = dict(plan=plan, next_group=next_group, total_groups=total_groups,
                 counts=[dict(sample=s, line=l, count=n) for (s, l), n in counts.items()])
    folder = root / "checkpoints"
    folder.mkdir(exist_ok=True)
    tmp = folder / "selection_checkpoint.zip.tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("state.json", json.dumps(state))
        archive.writestr("heaps.parquet", sink.getvalue().to_pybytes())
    tmp.replace(folder / "selection_checkpoint.zip")


def select(args):
    root = Path(args.out)
    plan, samples = describe(root)
    if (root / "selection_status.json").exists():
        status = json.loads((root / "selection_status.json").read_text())
        if status.get("complete_obs_scan"):
            if json.loads((root / "selection_plan.json").read_text()) != plan:
                raise SystemExit("Plan changed after selection. Use another output folder.")
            print("Complete selection already exists on Drive; reusing it.")
            return
    target = {r["sample"]: r for r in samples}
    caps = {r["sample"]: (plan["control_cells_per_sample"] if r["drug"] == CONTROL
                         else plan["cells_per_sample"]) for r in samples}
    heaps = defaultdict(list)
    counts = Counter()
    checkpoint = root / "checkpoints/selection_checkpoint.zip"
    start_group = 0
    if checkpoint.exists():
        with zipfile.ZipFile(checkpoint) as archive:
            state = json.loads(archive.read("state.json"))
            if state["plan"] != plan:
                raise SystemExit("Plan changed since the checkpoint. Use another output folder.")
            saved = pq.read_table(pa.BufferReader(archive.read("heaps.parquet"))).to_pydict()
        for s, l, b, priority in zip(saved["sample"], saved["cell_line_id"],
                                    saved["BARCODE_SUB_LIB_ID"], saved["priority"]):
            heaps[(s, l)].append((-priority, b))
        for heap in heaps.values():
            heapq.heapify(heap)
        counts.update({(r["sample"], r["line"]): r["count"] for r in state["counts"]})
        start_group = state["next_group"]
        print(f"Resuming selection at observation group {start_group + 1}.", flush=True)
    fs = HfFileSystem()
    cols = ["sample", "cell_line", "BARCODE_SUB_LIB_ID", "pass_filter", "gene_count", "pcnt_mito"]
    with open_remote(fs, plan["revision"], "metadata/obs_metadata.parquet") as source:
        pf = pq.ParquetFile(source)
        last_group = start_group
        for rg in range(start_group, pf.num_row_groups):
            if args.max_obs_groups and rg >= args.max_obs_groups:
                break
            for batch in pf.iter_batches(row_groups=[rg], columns=cols, batch_size=65536):
                table = pa.Table.from_batches([batch])
                mask = pc.and_(pc.is_in(table["sample"], value_set=pa.array(list(target))),
                               pc.is_in(table["cell_line"], value_set=pa.array(plan["cell_line_ids"])))
                mask = pc.and_(mask, pc.equal(table["pass_filter"], "full"))
                mask = pc.and_(mask, pc.greater_equal(table["gene_count"], plan["min_genes"]))
                mask = pc.and_(mask, pc.less_equal(table["pcnt_mito"], plan["max_mito_fraction"]))
                selected = table.filter(mask).select(["sample", "cell_line", "BARCODE_SUB_LIB_ID"])
                data = selected.to_pydict()
                for sample, line, barcode in zip(data["sample"], data["cell_line"], data["BARCODE_SUB_LIB_ID"]):
                    key = (sample, line)
                    counts[key] += 1
                    rank = int.from_bytes(hashlib.blake2b(
                        f"{plan['seed']}:{barcode}".encode(), digest_size=8).digest(), "big")
                    item = (-rank, barcode)
                    heap = heaps[key]
                    cap = caps[sample]
                    if len(heap) < cap:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
            print(f"Selected IDs: obs group {rg+1}/{pf.num_row_groups}", flush=True)
            last_group = rg + 1
            if last_group % getattr(args, "selection_checkpoint_every", 5) == 0 or last_group == pf.num_row_groups:
                selection_checkpoint(root, plan, heaps, counts, last_group, pf.num_row_groups)
                print("Selection checkpoint saved to Drive.", flush=True)
        scan_complete = last_group == pf.num_row_groups
        if not scan_complete:
            selection_checkpoint(root, plan, heaps, counts, last_group, pf.num_row_groups)
            print("Selection is partial. Rerun without --max-obs-groups to finish; export is disabled.")
            return
    rows = []
    report = []
    for sample in target:
        for line in plan["cell_line_ids"]:
            key = (sample, line)
            heap = heaps.get(key, [])
            report.append(dict(sample=sample, cell_line_id=line, eligible=counts[key], selected=len(heap)))
            rows.extend(dict(sample=sample, cell_line_id=line, BARCODE_SUB_LIB_ID=b) for _, b in heap)
    if not rows:
        raise SystemExit("No cells selected. Check plan and QC settings.")
    selection_tmp = root / "selected_cells.parquet.tmp"
    pq.write_table(pa.Table.from_pylist(rows), selection_tmp, compression="zstd")
    selection_tmp.replace(root / "selected_cells.parquet")
    save_json(root / "selection_report.json", report)
    # Pin the exact selection plan; edited plans cannot be mixed with old IDs.
    save_json(root / "selection_plan.json", plan)
    save_json(root / "selection_status.json", dict(selected=len(rows),
        complete_obs_scan=True, max_obs_groups=0))
    print(f"Saved {len(rows):,} selected IDs. Expression data has not been downloaded.", flush=True)


def may_contain_samples(pf, rg, sample_names):
    group = pf.metadata.row_group(rg)
    for column in range(group.num_columns):
        c = group.column(column)
        if c.path_in_schema == "sample":
            stat = c.statistics
            if stat is not None and stat.has_min_max:
                index = bisect.bisect_left(sample_names, stat.min)
                return index < len(sample_names) and sample_names[index] <= stat.max
    return True


def clean_row(row, metadata):
    genes, values = row["genes"], row["expressions"]
    if len(genes) != len(values):
        raise ValueError("Misaligned genes and expressions")
    if values and values[0] < 0:
        genes, values = genes[1:], values[1:]
    if any(v < 0 for v in values):
        raise ValueError("Unexpected negative count after removing marker")
    if any(g < 0 or g > 65535 for g in genes):
        raise ValueError("Gene token exceeds uint16 range")
    row["genes"], row["expressions"] = genes, values
    dose = ast.literal_eval(metadata["drugname_drugconc"])
    if len(dose) != 1 or dose[0][2] != "uM":
        raise ValueError("Unexpected dose representation")
    row["dose_uM"] = float(dose[0][1])
    return row


def download(args):
    root = Path(args.out)
    plan, samples = describe(root)
    if not (root / "selection_status.json").exists() or not json.loads(
            (root / "selection_status.json").read_text()).get("complete_obs_scan"):
        raise SystemExit("Complete the cell-selection step before exporting.")
    if json.loads((root / "selection_plan.json").read_text()) != plan:
        raise SystemExit("Plan changed after selection. Prepare and select in a new directory.")
    selected_table = pq.read_table(root / "selected_cells.parquet")
    wanted = set(selected_table["BARCODE_SUB_LIB_ID"].to_pylist())
    sample_meta = {r["sample"]: r for r in samples}
    sample_names = sorted(sample_meta)
    files = json.loads((root / "expression_files.json").read_text())
    out = root / "cells"
    out.mkdir(exist_ok=True)
    progress_path = root / "download_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    budget = int(plan["max_output_gb"] * 1_000_000_000)
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    fs = HfFileSystem()
    cols = ["genes", "expressions", "drug", "sample", "BARCODE_SUB_LIB_ID", "cell_line_id", "plate"]
    processed = 0
    for file_index, path in enumerate(files):
        if path in progress:
            continue
        if args.max_shards and processed >= args.max_shards:
            break
        part = out / f"part-{file_index:05d}.parquet"
        tmp = part.with_suffix(".parquet.tmp")
        # A successful rename followed by an interrupted checkpoint is recovered.
        if part.exists():
            table = pq.read_table(part, columns=["sample", "cell_line_id"])
            count = table.num_rows
            pairs = Counter(zip(table["sample"].to_pylist(), table["cell_line_id"].to_pylist()))
        else:
            rows = []
            with open_remote(fs, plan["revision"], path) as source:
                pf = pq.ParquetFile(source)
                rgs = [rg for rg in range(pf.num_row_groups) if may_contain_samples(pf, rg, sample_names)]
                if rgs:
                    identities = pf.read_row_groups(rgs, columns=["BARCODE_SUB_LIB_ID"])
                    offsets, offset = [], 0
                    for rg in rgs:
                        offsets.append(offset)
                        offset += pf.metadata.row_group(rg).num_rows
                    locations = defaultdict(list)
                    for i, barcode in enumerate(identities["BARCODE_SUB_LIB_ID"].to_pylist()):
                        if barcode in wanted:
                            block = bisect.bisect_right(offsets, i) - 1
                            locations[rgs[block]].append(i - offsets[block])
                    for rg, indices in locations.items():
                        table = pf.read_row_group(rg, columns=cols).take(pa.array(indices, type=pa.int64()))
                        rows.extend(clean_row(row, sample_meta[row["sample"]]) for row in table.to_pylist())
            count = len(rows)
            pairs = Counter((r["sample"], r["cell_line_id"]) for r in rows)
            if rows:
                schema = pa.schema([("genes", pa.list_(pa.uint16())),
                    ("expressions", pa.list_(pa.float32())), ("drug", pa.string()),
                    ("sample", pa.string()), ("BARCODE_SUB_LIB_ID", pa.string()),
                    ("cell_line_id", pa.string()), ("plate", pa.string()), ("dose_uM", pa.float32())])
                # Serialize in memory, measure BEFORE writing, and retain 100 MB
                # within the output limit for checkpoints and reports.
                sink = pa.BufferOutputStream()
                pq.write_table(pa.Table.from_pylist(rows, schema=schema), sink, compression="zstd")
                buffer = sink.getvalue()
                if used + buffer.size + 100_000_000 > budget or shutil.disk_usage(root).free < buffer.size + 100_000_000:
                    raise SystemExit("Output budget/free space reached; this shard was not written. "
                                     "Use fewer cells in a NEW plan, then rerun select/download.")
                with tmp.open("wb") as handle:
                    handle.write(buffer)
                tmp.replace(part)
                used += buffer.size
        progress[path] = dict(cells=count, strata=[dict(sample=s, cell_line_id=l, count=n)
                                                   for (s, l), n in pairs.items()])
        processed += 1
        if processed % getattr(args, "download_checkpoint_every", 25) == 0:
            save_json(progress_path, progress)
        print(f"Shard {file_index+1}/{len(files)}: saved {count:,}; "
              f"output about {used/1e9:.3f} GB", flush=True)
    save_json(progress_path, progress)
    saved_counts = Counter()
    for entry in progress.values():
        for row in entry["strata"]:
            saved_counts[(row["sample"], row["cell_line_id"])] += row["count"]
    coverage = []
    for row in json.loads((root / "selection_report.json").read_text()):
        coverage.append(dict(row, saved=saved_counts[(row["sample"], row["cell_line_id"])]))
    save_json(root / "coverage_report.json", coverage)
    saved = sum(saved_counts.values())
    complete = len(progress) == len(files)
    status = dict(complete_expression_scan=complete, processed_shards=len(progress),
                  total_shards=len(files), selected_cells=len(wanted), saved_cells=saved,
                  missing_selected_cells=len(wanted)-saved,
                  output_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()))
    save_json(root / "download_status.json", status)
    print(json.dumps(status, indent=2))
    if complete and saved != len(wanted):
        raise SystemExit("Some selected obs IDs are missing from expression data. Inspect coverage_report.json.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "select", "download", "describe"])
    parser.add_argument("--out", default="tahoe_subset")
    parser.add_argument("--lines", type=int, default=25)
    parser.add_argument("--drugs", type=int, default=100)
    parser.add_argument("--cells", type=int, default=100, help="per cell line and treatment sample")
    parser.add_argument("--controls", type=int, default=200, help="per cell line and DMSO sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--max-mito", type=float, default=0.20)
    parser.add_argument("--max-gb", type=float, default=35.0, help="decimal GB; leaves space within a 50 GB allowance")
    parser.add_argument("--max-obs-groups", type=int, default=0, help="SMOKE TEST ONLY; 0 scans all obs")
    parser.add_argument("--max-shards", type=int, default=0, help="limit export scan for a smoke test; rerun without limit to resume")
    args = parser.parse_args()
    if min(args.lines, args.drugs, args.cells, args.controls, args.max_gb) <= 0:
        parser.error("Counts and budget must be positive")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "select":
        select(args)
    elif args.command == "download":
        download(args)
    else:
        describe(Path(args.out))


if __name__ == "__main__":
    main()

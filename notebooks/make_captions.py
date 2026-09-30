"""Build captions.csv: one natural-language sentence per DeepFashion image.

Sources: Anno_coarse (category + 1000 noisy attributes, all 289,222 images)
         Anno_fine   (26 cleaner attributes, 20,000 images; overrides coarse where they overlap)
Only positive attributes (label == 1) are used, filtered through a hand-curated vocabulary
and category-compatibility rules so the sentence stays sensible for the garment.
"""
import csv
import numpy as np

ROOT = "."
attr_names = [l.split()[0] if len(l.split()) == 2 else " ".join(l.split()[:-1])
              for l in open(f"{ROOT}/Anno_coarse/list_attr_cloth.txt").read().splitlines()[2:]]
cats = [l.rsplit(None, 1) for l in open(f"{ROOT}/Anno_coarse/list_category_cloth.txt").read().splitlines()[2:]]
CAT_NAME = [c[0] for c in cats]
CAT_TYPE = {c[0]: int(c[1]) for c in cats}

# ---------------------------------------------------------------- category phrasing
CAT_PHRASE = {
    "Anorak": "anorak", "Blazer": "blazer", "Blouse": "blouse", "Bomber": "bomber jacket",
    "Button-Down": "button-down shirt", "Cardigan": "cardigan", "Flannel": "flannel shirt",
    "Halter": "halter top", "Henley": "henley shirt", "Hoodie": "hoodie", "Jacket": "jacket",
    "Jersey": "jersey", "Parka": "parka", "Peacoat": "peacoat", "Poncho": "poncho",
    "Sweater": "sweater", "Tank": "tank top", "Tee": "t-shirt", "Top": "top",
    "Turtleneck": "turtleneck", "Capris": "capri pants", "Chinos": "chinos", "Culottes": "culottes",
    "Cutoffs": "cutoff shorts", "Gauchos": "gaucho pants", "Jeans": "jeans",
    "Jeggings": "jeggings", "Jodhpurs": "jodhpurs", "Joggers": "joggers", "Leggings": "leggings",
    "Sarong": "sarong", "Shorts": "shorts", "Skirt": "skirt", "Sweatpants": "sweatpants",
    "Sweatshorts": "sweatshorts", "Trunks": "swim trunks", "Caftan": "caftan", "Cape": "cape",
    "Coat": "coat", "Coverup": "cover-up", "Dress": "dress", "Jumpsuit": "jumpsuit",
    "Kaftan": "kaftan", "Kimono": "kimono", "Nightdress": "nightdress", "Onesie": "onesie",
    "Robe": "robe", "Romper": "romper", "Shirtdress": "shirt dress", "Sundress": "sundress",
}

# ---------------------------------------------------------------- vocabulary
# name -> (slot, phrase, spec). spec: None = any garment; 'up'/'low'/'full' = category_type 1/2/3;
# a set of category names = only those; several may be combined in a tuple (union).
TABLE = {}
SLOT_ORDER = ["style", "struct", "fit", "color", "finish", "fabric", "sil", "len",
              "pat", "emb", "neck", "sleeve", "detail"]


def add(slot, phrase, names, spec=None):
    for n in names.split("|"):
        TABLE[n] = (slot, phrase, spec)


UP, LOW, FULL = "up", "low", "full"
DRESSY = {"Dress", "Caftan", "Kaftan", "Nightdress", "Shirtdress", "Sundress"}
LOWSTRAIGHT = {"Jeans", "Jeggings", "Chinos", "Joggers", "Leggings", "Sweatpants", "Culottes", "Capris", "Jodhpurs", "Gauchos"}
OUTER = {"Anorak", "Blazer", "Bomber", "Cardigan", "Jacket", "Parka", "Peacoat", "Poncho", "Coat", "Cape", "Kimono", "Robe", "Coverup"}
WINTER = OUTER | {"Turtleneck", "Sweater", "Hoodie", "Flannel", "Sweatpants", "Henley"}

# -- style (type 5), max 2
add("style", "classic", "classic")
add("style", "boho", "boho")
add("style", "chic", "chic")
add("style", "retro", "retro")
add("style", "elegant", "elegant")
add("style", "athletic", "athletic")
add("style", "sporty", "sporty")
add("style", "casual", "everyday")
add("style", "solid-colored", "solid")
add("style", "party", "party")
add("style", "beach", "beach")
add("style", "utility", "utility", (UP, FULL, "Shorts", "Chinos"))
add("style", "lounge", "lounge")
add("style", "yoga", "yoga", ("Leggings", "Tank", "Tee", "Top", "Joggers", "Shorts", "Sweatpants", "Hoodie"))
add("style", "workout", "workout", ("Leggings", "Tank", "Tee", "Top", "Joggers", "Shorts", "Sweatpants", "Hoodie", "Sweatshorts"))
add("style", "biker", "biker", (OUTER, "Shorts", "Leggings"))
add("style", "varsity", "varsity", (OUTER, "Jersey", "Tee", "Skirt"))
add("style", "summer", "summer", set(CAT_NAME) - WINTER)

# -- structure adjectives, max 2
add("struct", "sleeveless", "sleeveless", ({"Blouse", "Top", "Tee", "Dress", "Jumpsuit", "Romper", "Sundress", "Shirtdress", "Blazer", "Jacket", "Button-Down", "Jersey", "Kimono", "Coat", "Sweater"}, ))
add("struct", "strapless", "strapless", {"Top", "Dress", "Jumpsuit", "Romper", "Sundress", "Coverup", "Blouse"})
add("struct", "backless", "backless", {"Top", "Dress", "Jumpsuit", "Romper", "Sundress", "Blouse", "Halter", "Tank"})
add("struct", "hooded", "hooded|hood", {"Anorak", "Bomber", "Cardigan", "Jacket", "Parka", "Peacoat", "Poncho", "Coat", "Cape", "Robe", "Sweater", "Coverup", "Jersey", "Tee"})
add("struct", "collarless", "collarless", {"Blazer", "Jacket", "Cardigan", "Coat", "Top", "Blouse", "Bomber", "Peacoat", "Cape", "Kimono"})
add("struct", "racerback", "racerback", {"Tank", "Top", "Dress", "Jumpsuit", "Romper", "Sundress", "Halter"})
add("struct", "strappy", "strappy", {"Top", "Tank", "Dress", "Jumpsuit", "Romper", "Sundress", "Halter", "Coverup"})
add("struct", "off-the-shoulder", "off-the-shoulder|open-shoulder", {"Top", "Blouse", "Dress", "Romper", "Jumpsuit", "Sweater", "Sundress"})
add("struct", "one-shoulder", "one-shoulder", {"Top", "Dress", "Romper", "Jumpsuit"})
add("struct", "open-front", "open-front|knit open", {"Cardigan", "Jacket", "Blazer", "Kimono", "Coat", "Cape", "Poncho", "Robe", "Coverup", "Sweater", "Top"})
add("struct", "double-breasted", "double-breasted", {"Blazer", "Jacket", "Coat", "Peacoat", "Dress", "Bomber"})
add("struct", "asymmetrical", "asymmetrical|asymmetric")
add("struct", "cropped", "crop|cropped", (UP, "Jacket", LOWSTRAIGHT - {"Capris", "Leggings"}, "Jeans"))
add("struct", "cutout", "cutout|cutout-back|back cutout", (UP, FULL))

# -- fit / rise adjectives
add("fit", "high-waisted", "high-waisted|high-waist|high-rise|rise high-waisted", LOW)
add("fit", "low-rise", "low-rise", LOW)
add("fit", "mid-rise", "mid-rise|mid rise", LOW)
add("fit", "oversized", "oversized", (UP, FULL))
add("fit", "boxy", "boxy", UP)
add("fit", "slouchy", "slouchy", (UP, LOW))
add("fit", "fitted", "fitted", None)
add("fit", "longline", "longline", (UP, "Coat"))
add("fit", "relaxed-fit", "relaxed")
add("fit", "loose-fitting", "loose", None)
add("fit", "muscle-style", "muscle", {"Tee", "Tank", "Top", "Jersey"})

# -- color (type 5)
add("color", "red", "red")
add("color", "pink", "pink")

# -- finish / construction adjectives (max 2)
add("finish", "distressed", "distressed|destroyed", (LOW, "Jacket", "Tee", "Shorts"))
add("finish", "ripped", "ripped|shredded", (LOW, "Jacket"))
add("finish", "acid-washed", "acid wash|acid", (LOW, "Jacket"))
add("finish", "mineral-washed", "mineral wash|mineral", ("Tee", "Tank", "Top", "Sweatshirt", "Jersey", "Hoodie", "Shorts", "Dress"))
add("finish", "faded", "faded|bleached|bleach", (LOW, "Jacket", "Tee"))
add("finish", "washed", "wash|washed", (LOW, UP, "Dress", "Skirt"))
add("finish", "frayed", "frayed", (LOW, "Jacket", "Skirt"))
add("finish", "pleated", "pleated|pleat|pintuck pleated", None)
add("finish", "ruffled", "ruffled|ruffle", None)
add("finish", "tiered", "tiered", (DRESSY, "Skirt", "Top", "Blouse"))
add("finish", "layered", "layered", None)
add("finish", "quilted", "quilted", (OUTER, "Jacket", "Skirt", "Sweater"))
add("finish", "ribbed", "ribbed|rib|ribbed-knit|rib-knit", None)
add("finish", "cable-knit", "cable knit|cable-knit|cable", ("Sweater", "Cardigan", "Turtleneck", "Poncho", "Top"))
add("finish", "chunky", "chunky|chunky knit", ("Sweater", "Cardigan", "Turtleneck", "Poncho", "Top"))
add("finish", "textured", "textured")
add("finish", "heathered", "heathered", (UP, LOW))
add("finish", "marled", "marled", (UP,))
add("finish", "smocked", "smocked", (UP, FULL, "Skirt"))
add("finish", "neon", "neon")
add("finish", "metallic", "metallic")
add("finish", "tie-dye", "tie-dye|dip-dye|dip-dyed", None)

# -- fabrics (max 2)
for n in ["lace", "knit", "denim", "chiffon", "crochet", "cotton", "leather", "mesh", "woven", "sheer", "crepe",
          "satin", "velvet", "tweed", "tulle", "twill", "suede", "linen", "corduroy", "chambray", "gauze",
          "organza", "jacquard", "seersucker", "neoprene", "nylon", "fur", "brocade", "canvas", "georgette",
          "shearling", "velveteen", "sateen", "ponte", "scuba", "stretch", "eyelet", "flannel", "terry",
          "slub", "waffle", "burnout", "fuzzy"]:
    add("fabric", n, n)
add("fabric", "crochet", "crocheted|crochet knit")
add("fabric", "lace", "floral lace|crochet lace|eyelash lace|lacy|lace print")
add("fabric", "linen-blend", "linen-blend")
add("fabric", "cotton-blend", "cotton-blend")
add("fabric", "french terry", "french terry")
add("fabric", "faux leather", "faux leather")
add("fabric", "faux fur", "faux fur")
add("fabric", "faux suede", "faux suede")
add("fabric", "faux shearling", "faux shearling")
add("fabric", "sequined", "sequin|sequined")
add("fabric", "eyelash knit", "eyelash")
add("fabric", "loose-knit", "loose-knit")
add("fabric", "stretch-knit", "stretch-knit")
add("fabric", "slub-knit", "slub-knit")

# -- silhouette (max 1)
add("sil", "bodycon", "bodycon|bandage", DRESSY | {"Skirt"})
add("sil", "skater", "skater", DRESSY | {"Skirt"})
add("sil", "shift", "shift", DRESSY)
add("sil", "sheath", "sheath", DRESSY)
add("sil", "a-line", "a-line", DRESSY | {"Skirt"})
add("sil", "fit-and-flare", "fit flare", DRESSY)
add("sil", "swing", "swing", DRESSY | {"Top", "Coat", "Jacket"})
add("sil", "trapeze", "trapeze", DRESSY | {"Top", "Blouse"})
add("sil", "tulip", "tulip", DRESSY | {"Skirt"})
add("sil", "slip", "slip", DRESSY | {"Skirt"})
add("sil", "wrap", "wrap|faux-wrap", DRESSY | {"Skirt", "Top", "Blouse", "Sweater", "Cardigan", "Jumpsuit", "Romper"})
add("sil", "pencil", "pencil", {"Skirt"})
add("sil", "peasant", "peasant", DRESSY | {"Blouse", "Top", "Tee", "Tank"})
add("sil", "tunic", "tunic", DRESSY | {"Top", "Blouse", "Sweater", "Tee"})
add("sil", "cami", "cami", DRESSY | {"Top", "Romper", "Jumpsuit"})
add("sil", "skinny", "skinny|rise skinny|mid-rise skinny", {"Jeans", "Jeggings", "Chinos", "Joggers", "Leggings"})
add("sil", "slim", "slim", {"Jeans", "Jeggings", "Chinos", "Joggers", "Blazer", "Jacket", "Button-Down", "Tee"})
add("sil", "flared", "flare|flared", {"Jeans", "Jeggings", "Leggings", "Skirt", "Dress"})
add("sil", "wide-leg", "wide-leg", {"Jeans", "Culottes", "Gauchos", "Jumpsuit", "Sweatpants", "Chinos"})
add("sil", "straight-leg", "straight-leg", {"Jeans", "Chinos", "Jeggings"})
add("sil", "harem", "harem", {"Joggers", "Sweatpants", "Jumpsuit", "Chinos", "Culottes"})
add("sil", "boyfriend", "boyfriend", {"Jeans", "Tee", "Blazer", "Cardigan", "Jacket", "Sweater", "Button-Down"})
add("sil", "cargo", "cargo", {"Shorts", "Chinos", "Joggers", "Jacket", "Capris", "Sweatpants"})
add("sil", "bermuda", "bermuda", {"Shorts", "Sweatshorts"})
add("sil", "moto", "moto", {"Jacket", "Leggings", "Jeans", "Jeggings"})
add("sil", "puffer", "puffer", (OUTER,))
add("sil", "polo", "polo", {"Top", "Tee", "Dress", "Jersey"})
add("sil", "tube", "tube", {"Top", "Dress"})
add("sil", "high-low", "high-low", DRESSY | {"Skirt", "Top", "Blouse", "Tee", "Tank"})
add("sil", "trouser", "trouser", {"Shorts", "Chinos"})

# -- length (max 1)
add("len", "maxi", "maxi", DRESSY | {"Skirt"})
add("len", "midi", "midi", DRESSY | {"Skirt"})
add("len", "mini", "mini", DRESSY | {"Skirt"})
add("len", "knee-length", "knee-length", DRESSY | {"Skirt"})

# -- patterns / prints (with-clause)
add("pat", "a floral print", "floral|floral print|floral pattern|flower|ditsy floral|ditsy floral print|ditsy|floral paisley|abstract floral|abstract floral print|floral textured|floral flutter|botanical|botanical print|wildflower|sunflower|garden|roses|rose print")
add("pat", "stripes", "striped|stripe|stripes|pinstripe|pinstriped|multi-stripe|classic striped|rugby stripe|rugby striped|mixed stripe|breton|breton stripe|nautical striped|nautical stripe|varsity-striped|knit striped|knit stripe|marled stripe|heathered stripe|ribbed stripe|abstract stripe|geo stripe|boxy striped|vertical")
add("pat", "polka dots", "polka dot|dot|dots|dotted|spotted|speckled")
add("pat", "an abstract print", "abstract|abstract print|abstract printed|abstract pattern")
add("pat", "a paisley print", "paisley|paisley print|abstract paisley|ornate paisley")
add("pat", "a geometric print", "geo|geo print|geo pattern|abstract geo|abstract geo print|grid|grid print|diamond|diamond print|abstract diamond|zigzag|zig")
add("pat", "a chevron print", "chevron|chevron print|abstract chevron|abstract chevron print")
add("pat", "a leopard print", "leopard|leopard print|cheetah")
add("pat", "an animal print", "animal|animal print|giraffe|giraffe print|snake")
add("pat", "a tribal print", "tribal|tribal-inspired|folk|folk print|mandala|mandala print|medallion|medallion print|ikat|ikat print")
add("pat", "a southwestern print", "southwestern|southwestern-patterned|southwestern-inspired|southwestern-print")
add("pat", "a baroque print", "baroque|baroque print|ornate|ornate print|kaleidoscope|kaleidoscope print|mosaic|mosaic print")
add("pat", "a heart print", "heart|heart print")
add("pat", "a daisy print", "daisy|daisy print")
add("pat", "a palm print", "palm|palm print|palm tree|palm springs")
add("pat", "a leaf print", "leaf|leaf print|leave")
add("pat", "a butterfly print", "butterfly|butterfly print")
add("pat", "a bird print", "bird|bird print")
add("pat", "a bandana print", "bandana|bandana print")
add("pat", "a camouflage print", "camo|camouflage")
add("pat", "a colorblock design", "colorblock|colorblocked|two-tone")
add("pat", "an ombre effect", "ombre")
add("pat", "a watercolor print", "watercolor|brushstroke|brushstroke print|paint|painted|paint splatter|splatter")
add("pat", "a plaid pattern", "plaid|tartan|plaid shirt|checked|checkered|windowpane")
add("pat", "a gingham pattern", "gingham")
add("pat", "a houndstooth pattern", "houndstooth")
add("pat", "a herringbone pattern", "herringbone")
add("pat", "a nautical theme", "nautical")
add("pat", "a tropical print", "tropical")
add("pat", "a marble print", "marble|marble print")
add("pat", "a graphic print", "graphic")
add("pat", "a lattice pattern", "lattice")
add("pat", "a logo", "logo")
add("pat", "a printed design", "print|printed|pattern|patterned")  # generic; dropped if anything more specific

# -- embellishment / trim (with-clause)
add("emb", "embroidery", "embroidered|embroidery|embroidered floral|floral-embroidered|embroidered lace|embroidered mesh|embroidered gauze|embroidered woven")
add("emb", "beading", "beaded|bead|beaded chiffon|beaded sheer|beaded collar")
add("emb", "sequins", "sequin|sequined")
add("emb", "rhinestones", "rhinestone|bejeweled|gem")
add("emb", "embellishments", "embellished")
add("emb", "studs", "studded", None)
add("emb", "appliques", "applique")
add("emb", "fringe", "fringe|fringed|crochet fringe")
add("emb", "tassels", "tassel|tasseled|tasseled")
add("emb", "ruffles", "ruffle trim|flounce|flounced|flutter")
add("emb", "lace trim", "lace trim|lace-trim|lace-trimmed|lace trimmed")
add("emb", "contrast trim", "contrast trim|contrast|contrast-trimmed")
add("emb", "lace panels", "lace panel|lace-paneled|lace paneled|lace overlay|lace layered")
add("emb", "mesh panels", "mesh panel|mesh-paneled|mesh paneled|mesh-trimmed|mesh overlay")
add("emb", "leather trim", "leather-trimmed|leather trimmed|leather-paneled|leather paneled|faux leather-trimmed|faux leather-paneled|faux leather paneled")
add("emb", "chiffon panels", "chiffon-paneled|chiffon paneled")
add("emb", "crochet panels", "crochet-paneled|crochet-trimmed")
add("emb", "sheer panels", "sheer-paneled")
add("emb", "eyelet detailing", "eyelet", None)
add("emb", "pintucks", "pintuck|pintucked")
add("emb", "a bow", "bow|bow-front|bow-back|back bow", None)

# -- neckline (max 1) and sleeves (max 1): garment types with upper body
NECK = (UP, FULL)
add("neck", "a v-neckline", "v-neck|striped v-neck|knit v-neck|print v-neck|heathered v-neck|fitted v-neck|classic v-neck|v-cut", NECK)
add("neck", "a deep v-neckline", "deep v-neck|deep-v", NECK)
add("neck", "a crew neckline", "crew|crew neck|classic crew|classic crew neck", NECK)
add("neck", "a scoop neckline", "scoop|scoop-neck", NECK)
add("neck", "a surplice neckline", "surplice|print surplice|floral surplice|floral print surplice|chiffon surplice|draped surplice", NECK)
add("neck", "a cowl neckline", "cowl|cowl neck", NECK)
add("neck", "a boat neckline", "boat neck", NECK)
add("neck", "a mock neck", "mock|mock neck|mock-neck", NECK)
add("neck", "a high neck", "turtle-neck|high-neck|tie-neck", NECK)
add("neck", "a keyhole neckline", "keyhole", NECK)
add("neck", "a shawl collar", "shawl|knit shawl|draped shawl", NECK)
add("neck", "a notched collar", "notched collar|lapel", NECK)
add("neck", "a square neckline", "square-neckline", NECK)
add("neck", "a collar", "collar|collared|arrow collar|collar lace", NECK)
add("neck", "no neckline detail", "no_neckline", NECK)  # placeholder, removed below
del TABLE["no_neckline"]

NOSLEEVE = {"Tank", "Halter", "Cape", "Poncho", "Coverup", "Kimono"}
add("sleeve", "long sleeves", "long sleeve|long-sleeve|long-sleeved", (UP, FULL))
add("sleeve", "short sleeves", "short_sleeve", (UP, FULL))
add("sleeve", "cap sleeves", "cap-sleeve", (UP, FULL))
add("sleeve", "dolman sleeves", "dolman|dolman sleeve|dolman-sleeve", (UP, FULL))
add("sleeve", "raglan sleeves", "raglan|raglan sleeve|knit raglan", (UP, FULL))
add("sleeve", "batwing sleeves", "batwing", (UP, FULL))
add("sleeve", "bell sleeves", "bell|bell-sleeve", (UP, FULL))
add("sleeve", "flutter sleeves", "flutter sleeve|flutter-sleeve", (UP, FULL))
add("sleeve", "drop sleeves", "drop-sleeve", (UP, FULL))

# -- other parts (with-clause)
add("detail", "pockets", "pocket|colorblock pocket|classic pocket|boxy pocket|knit pocket|zip-pocket")
add("detail", "a kangaroo pocket", "kangaroo|kangaroo pocket", (UP, FULL))
add("detail", "buttons", "button|buttoned|button-front|two-button|one-button|single-button|toggle|snap")
add("detail", "a zipper", "zip|zipper|zippered|zipped|zip-front|zip-up")
add("detail", "a drawstring", "drawstring|denim drawstring|cotton drawstring|chambray drawstring")
add("detail", "a belt", "belted|belted maxi|belted floral|belted lace|belted chiffon|belted plaid|belted floral print")
add("detail", "a peplum", "peplum|lace peplum|leather peplum", (UP, FULL))
add("detail", "a slit", "slit|side slit|side-slit|high-slit|m-slit|vented hem|vent|split", ({"Skirt"}, DRESSY, "Jumpsuit", "Coat"))
add("detail", "a drop waist", "drop waist|drop-waist|dropped", (DRESSY, "Skirt"))
add("detail", "a gathered waist", "gathered waistline|cinched|ruched|shirred|elasticized")
add("detail", "a lace-up detail", "lace-up")
add("detail", "a tie front", "tie-front|twist-front|drape-front|self-tie", (UP, FULL))
add("detail", "a cuffed hem", "cuffed", (LOW,))
add("detail", "a curved hem", "curved hem|curved|scalloped|scallop", None)
add("detail", "suspender straps", "suspender", None)
add("detail", "a hood", "hooded utility|hooded maxi", None)
del TABLE["hooded utility"], TABLE["hooded maxi"]

# Fine annotation names -> (slot, phrase, spec); mapped onto the same slots.
FINE = ["floral", "graphic", "striped", "embroidered", "pleated", "solid", "lattice",
        "long_sleeve", "short_sleeve", "sleeveless", "maxi_length", "mini_length", "no_dress",
        "crew_neckline", "v_neckline", "square_neckline", "no_neckline",
        "denim", "chiffon", "cotton", "leather", "faux", "knit", "tight", "loose", "conventional"]
FINE_MAP = {
    "floral": "floral", "graphic": "graphic", "striped": "striped", "embroidered": "embroidered",
    "pleated": "pleated", "lattice": "lattice", "long_sleeve": "long sleeve",
    "short_sleeve": "short_sleeve", "sleeveless": "sleeveless", "maxi_length": "maxi",
    "mini_length": "mini", "crew_neckline": "crew", "v_neckline": "v-neck",
    "square_neckline": "square-neckline", "denim": "denim", "chiffon": "chiffon",
    "leather": "leather", "knit": "knit", "faux": "faux",
}
# Fine annotations are forced 1-of-N choices per group, so the catch-all options ("solid" texture,
# "cotton" fabric, "conventional" fit, "no_dress", "no_neckline") carry no information and are not
# mapped; the coarse annotations still contribute them when they explicitly list "solid"/"cotton".
# extra fine-only vocabulary entries
add("neck", "a square neckline", "square-neckline", NECK)
add("style", "solid-colored", "solid")


def spec_ok(spec, cat):
    if spec is None:
        return True
    ctype = CAT_TYPE[cat]
    if isinstance(spec, str):
        spec = (spec,)
    if isinstance(spec, set):
        spec = (spec,)
    for s in spec:
        if isinstance(s, str):
            if s == "up" and ctype == 1 or s == "low" and ctype == 2 or s == "full" and ctype == 3:
                return True
            if s == cat:
                return True
        elif cat in s:
            return True
    return False


PLURAL = {"Capris", "Chinos", "Culottes", "Cutoffs", "Gauchos", "Jeans", "Jeggings", "Jodhpurs", "Joggers",
          "Leggings", "Shorts", "Sweatpants", "Sweatshorts", "Trunks"}


def article(word):
    return "an" if word[0] in "aeiou" and not word.startswith("u") else "a"


def join_and(items):
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def build_sentence(cat, coarse_attrs, fine_attrs):
    """cat: category name; coarse_attrs / fine_attrs: lists of raw positive attribute names."""
    slots = {s: [] for s in SLOT_ORDER}

    def take(name, from_fine=False):
        if name not in TABLE:
            return
        slot, phrase, spec = TABLE[name]
        if not spec_ok(spec, cat):
            return
        if phrase not in [p for p, _ in slots[slot]]:
            slots[slot].append((phrase, from_fine))

    fine_mapped = [FINE_MAP[a] for a in fine_attrs if a in FINE_MAP]
    for a in coarse_attrs:
        take(a)
    for a in fine_mapped:
        take(a, True)

    # fine annotations are cleaner: where fine speaks on an exclusive slot, it wins over coarse
    for s in ("neck", "len"):
        fine_items = [p for p in slots[s] if p[1]]
        if fine_items:
            slots[s] = fine_items
    if len(slots["len"]) > 1 and not any(p[1] for p in slots["len"]):
        slots["len"] = []  # coarse maxi + mini contradict
    # sleeves: fine (long/short/sleeveless) wins; otherwise drop contradictory coarse combos
    fine_sleeve = [a for a in fine_attrs if a in ("long_sleeve", "short_sleeve", "sleeveless")]
    if fine_sleeve:
        slots["sleeve"] = [p for p in slots["sleeve"] if p[1]]
        slots["struct"] = [p for p in slots["struct"] if p[0] != "sleeveless" or "sleeveless" in fine_sleeve]
    if any(p[0] == "sleeveless" for p in slots["struct"]):
        slots["sleeve"] = [p for p in slots["sleeve"] if p[0].startswith(("dolman", "raglan")) is False and False]
    elif any(p[0] == "long sleeves" for p in slots["sleeve"]) and any(
            p[0] in ("short sleeves", "cap sleeves") for p in slots["sleeve"]):
        slots["sleeve"] = []
    # mutual exclusions
    if cat in NOSLEEVE:
        slots["sleeve"] = []
        slots["struct"] = [p for p in slots["struct"] if p[0] != "sleeveless"]
    if cat in ("Hoodie",):
        slots["struct"] = [p for p in slots["struct"] if p[0] != "hooded"]
    if len(slots["neck"]) > 1:
        slots["neck"] = slots["neck"][:1]
    if any(p[0] == "hooded" for p in slots["struct"]):
        slots["neck"] = []
    if any(p[0] in ("strapless", "off-the-shoulder", "one-shoulder") for p in slots["struct"]):
        slots["neck"] = []
        slots["sleeve"] = []

    # fabric: resolve faux, cap at 2
    fab = [p[0] for p in slots["fabric"]]
    if "faux" in [a for a in fine_attrs]:
        for base in ("leather", "fur", "suede", "shearling"):
            if base in fab and "faux " + base not in fab:
                fab = ["faux " + base if f == base else f for f in fab]
    # coarse 'faux' + base fabric
    if "faux" in coarse_attrs:
        for base in ("leather", "fur", "suede", "shearling"):
            if base in fab and "faux " + base not in fab:
                fab = ["faux " + base if f == base else f for f in fab]
    fab = [f for f in fab if not (f in ("leather", "fur", "suede", "shearling") and "faux " + f in fab)]
    fab = list(dict.fromkeys(fab))
    # drop redundancy with category
    if cat == "Flannel":
        fab = [f for f in fab if f != "flannel"]
    if cat == "Jeans":
        fab = [f for f in fab if f != "denim"] + (["denim"] if "denim" in fab else [])
    # knit + a more specific knit type; lace + crochet keep
    if "denim" in fab and len(fab) > 1:
        fab = [f for f in fab if f not in ("chiffon", "leather", "lace", "velvet", "satin", "tulle", "organza")]
    if "denim" in fab and len(fab) > 1:
        fab = [f for f in fab if f not in ("mesh", "knit", "crochet", "fur", "suede")]
    if "french terry" in fab:
        fab = [f for f in fab if f != "terry"]
    fab = fab[:2]

    # patterns: generic print only when nothing specific
    pats = [p[0] for p in slots["pat"]]
    if len(pats) > 1 and "a printed design" in pats:
        pats.remove("a printed design")
    pats = pats[:3]
    emb = [p[0] for p in slots["emb"]]
    fabset = set(fab)
    if "a floral print" in pats and "embroidery" in emb:
        pats.remove("a floral print")
        emb[emb.index("embroidery")] = "floral embroidery"
    # do not repeat fabric as trim
    if "lace" in fabset:
        emb = [e for e in emb if e not in ("lace panels",)] if False else emb
    if "leather" in fabset or "faux leather" in fabset:
        pass
    if "sequined" in fabset:
        emb = [e for e in emb if e != "sequins"]
    if "sheer" in fabset:
        emb = [e for e in emb if e != "sheer panels"]
    finish = [p[0] for p in slots["finish"]]
    if "pleated" in finish and cat == "Skirt" and False:
        pass
    if "acid-washed" in finish or "mineral-washed" in finish:
        finish = [f for f in finish if f not in ("washed", "faded")]
    if "distressed" in finish and "ripped" in finish:
        finish.remove("ripped")
    if "pleated" in finish and any("pleat" in e for e in emb):
        finish.remove("pleated")
    if "embroidered" in finish:
        finish.remove("embroidered")
    if "textured" in finish and ("cable-knit" in finish or "chunky" in finish):
        finish.remove("textured")
    if "cable-knit" in finish and "knit" in fab:
        fab.remove("knit")
        finish = [f for f in finish]
    if "tie-dye" in finish and "a printed design" in pats:
        pats = [p for p in pats if p != "a printed design"]
    if "heathered" in finish and "marled" in finish:
        finish.remove("marled")
    finish = finish[:2]
    if "tie-dye" in finish:
        finish = ["tie-dye" if f == "tie-dye" else f for f in finish]

    style = [p[0] for p in slots["style"]]
    if "solid-colored" in style and (pats or emb):
        style.remove("solid-colored")
    if "solid-colored" in style:
        style = ["solid-colored"] + [s for s in style if s != "solid-colored"]
    style = style[:2]
    struct = [p[0] for p in slots["struct"]]
    if "cutout" in struct and cat in ("Halter",):
        struct.remove("cutout")
    struct = struct[:2]
    fit = [p[0] for p in slots["fit"]]
    fit_rise = [f for f in fit if f in ("high-waisted", "low-rise", "mid-rise")][:1]
    fit_other = [f for f in fit if f not in ("high-waisted", "low-rise", "mid-rise")]
    if "loose-fitting" in fit_other and "fitted" in fit_other:
        fit_other.remove("loose-fitting"); fit_other.remove("fitted")
    if "boxy" in fit_other and "oversized" in fit_other:
        fit_other.remove("boxy")
    fit_other = fit_other[:1]
    if "tight" in [a for a in fine_attrs]:
        if cat not in ("Leggings", "Jeggings") and "loose-fitting" not in fit_other and "fitted" not in fit_other:
            fit_other = ["form-fitting"] if not fit_other else fit_other
    if "loose" in [a for a in fine_attrs] and not fit_other and "loose-fitting" not in fit_other:
        fit_other = ["loose-fitting"]
    color = [p[0] for p in slots["color"]]
    sil = [p[0] for p in slots["sil"]][:1]
    length = [p[0] for p in slots["len"]][:1]
    if sil and cat in ("Skirt",) and sil[0] == "pencil" and length and length[0] == "maxi":
        length = []
    if sil and sil[0] == "fit-and-flare" and length:
        pass

    adj = []
    adj += style
    adj += struct
    adj += fit_rise + fit_other
    if color:
        adj.append(" and ".join(color))
    adj += finish
    adj += fab
    adj += sil
    adj += length
    # avoid repeating the same word twice in the chain
    seen, adj2 = set(), []
    for a in adj:
        if a in seen:
            continue
        seen.add(a)
        adj2.append(a)
    adj = adj2

    noun = CAT_PHRASE[cat]
    head = " ".join(adj + [noun]) if adj else noun
    if cat in PLURAL:
        prefix = f"This is a pair of {head}"
    else:
        prefix = f"This is {article(head)} {head}"

    with_items = []
    for x in pats:
        with_items.append(x)
    for x in emb:
        with_items.append(("an " if x[0] in "aeiou" else "") + x if x in ("embellishments", "appliques") and False else x)
    neck = [p[0] for p in slots["neck"]]
    sleeve = [p[0] for p in slots["sleeve"]]
    with_items += neck + sleeve
    with_items += [p[0] for p in slots["detail"]]
    # dedupe & tidy
    with_items = list(dict.fromkeys(with_items))
    with_items = with_items[:5]
    if with_items:
        return prefix + " with " + join_and(with_items) + "."
    return prefix + "."


def main():
    coarse_paths, cat_of, coarse_pos = [], [], []
    with open(f"{ROOT}/Anno_coarse/list_category_img.txt") as f:
        next(f); next(f)
        for l in f:
            p, c = l.split()
            coarse_paths.append(p); cat_of.append(CAT_NAME[int(c) - 1])
    with open(f"{ROOT}/Anno_coarse/list_attr_img.txt") as f:
        next(f); next(f)
        for i, l in enumerate(f):
            p, rest = l.split(None, 1)
            assert p == coarse_paths[i]
            v = np.array(rest.split(), dtype=np.int8)
            coarse_pos.append([attr_names[j] for j in np.flatnonzero(v == 1)])

    conv = lambda p: "img/" + p.split("/")[1].rsplit("-img_", 1)[0] + "/img_" + p.rsplit("-img_", 1)[1]
    fine = {}
    with open(f"{ROOT}/Anno_fine/list_attr_img.txt") as f:
        next(f); next(f)
        for l in f:
            p, rest = l.split(None, 1)
            v = rest.split()
            fine[conv(p)] = [FINE[j] for j, x in enumerate(v) if x == "1"]

    with open(f"{ROOT}/captions.csv", "w", newline="") as out:
        w = csv.writer(out)
        w.writerow(["image_path", "caption"])
        for p, c, a in zip(coarse_paths, cat_of, coarse_pos):
            w.writerow([p, build_sentence(c, a, fine.get(p, []))])


if __name__ == "__main__":
    main()

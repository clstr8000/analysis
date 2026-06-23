# Lexikon erstellen (Version 6 – Themenlexikon ohne Sentiment, mit Cluster-Namen und 1–3-Wort-Ausdrücken)

# 0.0: Packages importieren
import pandas as pd  # (pandas für Tabellen, Datenverwendung)
import numpy as np   # (numpy auch für Tabellen, Datenverwendung)
from collections import Counter  # (counter um Worthäufigkeiten zu zählen)
from sentence_transformers import SentenceTransformer  # (enthält BERT)
from pathlib import Path

# 0.1: Konfiguration
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "yelp_final.csv"
OUTPUT_FILE = BASE_DIR / "topic_lexicon_seeds_only.xlsx"

TEXT_COLUMN = "text"
MAX_ROWS = 200_000
RANDOM_STATE = 42
MIN_TERM_FREQ = 5
MAX_CANDIDATES = 10_000
NGRAM_MAX_N = 3
MIN_TOPIC_CONFIDENCE = 0.7
EMBEDDING_BATCH_SIZE = 64

# 0.2: BERT laden
bert = SentenceTransformer("all-MiniLM-L6-v2")  # (BERT analysiert Text kontextuell)

# 1: Themenwörter zur Orientierung (Seeds)
TOPIC_SEEDS = {
    "FOOD": [
        "food", "burger", "sandwich", "chicken", "fries", "nuggets",
        "meal", "taste", "flavor", "sauce", "portion", "fresh",
        "delicious", "tasty", "crispy", "juicy", "spicy", "bland",
        "soggy", "cold food", "hot food", "overcooked", "undercooked",
        "burnt", "salty", "seasoning", "texture", "quality", "freshness",
        "temperature", "lukewarm", "dry chicken", "tender chicken",
        "chick-fil-a chicken biscuit", "spicy chicken biscuit",
        "chick-fil-a chick-n-minis", "chick-n-minis", "egg white grill",
        "hash brown scramble burrito", "hash brown scramble bowl",
        "chicken egg cheese biscuit", "bacon egg cheese biscuit",
        "sausage egg cheese biscuit", "chicken egg cheese muffin",
        "bacon egg cheese muffin", "sausage egg cheese muffin", "hash browns",
        "berry parfait", "fruit cup", "honey pepper pimento sandwich",
        "honey pepper pimento", "pimento cheese", "pickled jalapenos",
        "chick-fil-a chicken sandwich", "spicy deluxe sandwich",
        "spicy chicken sandwich", "chick-fil-a nuggets", "grilled nuggets",
        "chick-fil-a deluxe sandwich", "chick-fil-a cool wrap", "cool wrap",
        "chick-n-strips", "chicken strips", "grilled chicken sandwich",
        "grilled chicken club sandwich", "cobb salad", "spicy southwest salad",
        "market salad", "side salad", "waffle potato fries", "waffle fries",
        "mac and cheese", "mac & cheese", "chicken noodle soup",
        "waffle potato chips", "original flavor waffle potato chips",
        "chick-fil-a sauce flavored waffle potato chips", "apple sauce",
        "buddy fruits apple sauce", "kale crunch side", "peach milkshake",
        "vanilla milkshake", "cookies cream milkshake",
        "cookies & cream milkshake", "chocolate milkshake",
        "strawberry milkshake", "icedream cup", "icedream cone",
        "chocolate chunk cookie", "chocolate fudge brownie",
        "barbeque sauce", "chick-fil-a sauce", "garden herb ranch sauce",
        "honey mustard sauce", "polynesian sauce", "sweet spicy sriracha sauce",
        "sweet & spicy sriracha sauce", "zesty buffalo sauce",
        "honey roasted bbq sauce", "avocado lime ranch dressing",
        "creamy salsa dressing", "fat-free honey mustard dressing",
        "garden herb ranch dressing", "light balsamic vinaigrette dressing",
        "light italian dressing", "zesty apple cider vinaigrette dressing"
    ],

    "SERVICE": [
        "service", "staff", "employee", "employees", "worker", "workers",
        "cashier", "manager", "friendly", "helpful", "polite", "rude",
        "attentive", "unprofessional", "customer service", "greeted",
        "welcoming", "attitude", "respectful", "disrespectful",
        "kind", "courteous", "patient", "impatient", "professional",
        "pleasant", "smiling", "smile", "care", "caring", "hospitality",
        "team member", "crew", "server", "assistance", "help", "ignored",
        "apologized", "accommodating", "knowledgeable", "training"
    ],

    "SPEED": [
        "fast", "slow", "quick", "quickly", "speed", "speedy",
        "service speed", "order ready", "ready fast", "took forever",
        "took too long", "delay", "delayed", "immediate", "efficient",
        "inefficient", "wait", "waiting", "wait time", "long wait",
        "short wait", "line", "queue", "long line", "short line",
        "stood in line", "standing in line", "crowded line", "busy", "rush",
        "lunch rush", "dinner rush", "peak hour", "served quickly",
        "served fast", "ready quickly", "slow service", "fast service",
        "order time", "pickup time", "turnaround", "backed up"
    ],

    "HYGIENE": [
        "clean", "dirty", "messy", "filthy", "spotless", "sanitary",
        "unsanitary", "hygiene", "cleanliness", "sticky", "smell",
        "odor", "trash", "garbage", "bathroom", "restroom",
        "table", "tables", "floor", "floors", "counter", "counters",
        "sink", "toilet", "napkin", "spill", "spilled", "greasy",
        "dusty", "stain", "stained", "cleaned", "unclean", "neat",
        "tidy", "sanitized", "soap", "paper towel", "overflowing trash"
    ],

    "VALUE": [
        "price", "prices", "expensive", "cheap", "value", "worth",
        "deal", "combo", "meal deal", "affordable", "overpriced",
        "cost", "charged", "fair price", "reasonable", "unreasonable",
        "portion size", "money", "receipt", "bill", "total", "discount",
        "coupon", "reward", "rewards", "points", "free item", "promotion",
        "special", "pricey", "low price", "high price", "good value",
        "bad value", "not worth", "worth it", "small portion", "large portion"
    ],

    "AMBIENCE": [
        "ambience", "atmosphere", "vibe", "environment", "seating",
        "dining room", "inside", "restaurant", "location", "crowded",
        "quiet", "loud", "noisy", "comfortable", "uncomfortable",
        "decor", "music", "lighting", "space", "layout", "interior",
        "booth", "chair", "chairs", "table", "tables", "family friendly",
        "kid friendly", "play area", "temperature", "air conditioning",
        "warm", "cold", "busy atmosphere", "calm", "relaxing", "modern"
    ],

    "PARKING": [
        "parking", "parking lot", "parking spot", "parking spaces",
        "car", "cars", "lot", "garage", "parked", "hard to park",
        "easy parking", "traffic", "entrance", "exit", "street parking",
        "curbside", "curb", "pickup spot", "parking area", "small lot",
        "full lot", "crowded lot", "traffic flow", "turn in", "turn out",
        "blocked", "congested", "nearby parking", "free parking"
    ],

    "DRINKS": [
        "drink", "drinks", "beverage", "soda", "tea", "sweet tea",
        "iced tea", "lemonade", "water", "coffee", "milkshake",
        "shake", "refill", "fountain drink", "ice", "watery",
        "cold drink", "hot drink", "carbonated", "flat soda", "fresh lemonade",
        "pineapple dragonfruit sprite", "pineapple dragonfruit & sprite",
        "pineapple dragonfruit lemonade", "pineapple dragonfruit sunjoy",
        "pineapple dragonfruit teas", "chick-fil-a lemonade", "diet lemonade",
        "sunjoy", "half sweet tea half lemonade",
        "half sweet tea half diet lemonade", "half unsweet tea half lemonade",
        "half unsweet tea half diet lemonade", "freshly-brewed sweetened iced tea",
        "freshly-brewed unsweetened iced tea", "sweetened iced tea",
        "unsweetened iced tea", "iced coffee", "dasani bottled water",
        "bottled water", "honest kids apple juice", "apple juice",
        "simply orange", "orange juice", "chocolate milk", "1% chocolate milk",
        "milk", "1% milk", "dr pepper", "seasonal gallon beverages",
        "gallon beverages", "coca-cola", "hot coffee",
        "pineapple dragonfruit frosted lemonade", "peach frosted lemonade",
        "frosted lemonade", "frosted sodas", "floats", "frosted coffee"
    ],

    "ACCESSIBILITY": [
        "wheelchair", "accessible", "accessibility", "entrance",
        "ramp", "stairs", "disabled", "handicap", "elevator",
        "restroom access", "parking access", "easy access", "door",
        "automatic door", "wide door", "narrow door", "mobility", "walker",
        "stroller", "step", "curb", "sidewalk", "path", "space",
        "accessible parking", "handicap parking", "accessible table",
        "accessible restroom", "low counter", "high counter"
    ],

    "GENERAL": [
        "good", "bad", "great", "nice", "amazing", "terrible",
        "awful", "excellent", "fine", "okay", "average",
        "experience", "visit", "place", "spot", "location",
        "recommend", "disappointed", "satisfied", "overall", "favorite",
        "best", "worst", "love", "liked", "enjoyed", "happy", "unhappy",
        "return", "come back", "never again", "consistent", "inconsistent"
    ]
}

# 2: N-Gramme extrahieren (1–3 Wörter)
def extract_ngrams(text, n=3):
    tokens = [t.lower() for t in text.split() if t.replace("-", "").isalpha()]
    ngrams = []

    # 1-Wort-Ausdrücke
    for t in tokens:
        if len(t) > 2:
            ngrams.append(t)

    # 2- und 3-Wort-Ausdrücke
    for size in range(2, n+1):
        for i in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[i:i+size])
            if all(len(w) > 2 for w in tokens[i:i+size]):
                ngrams.append(phrase)

    return ngrams

# 3: Wörter aus dem Text filtern
def build_candidate_words(df):
    print("Extrahiere Kandidaten aus Review-Texten...")
    all_words = []
    for text in df[TEXT_COLUMN]:
        if not isinstance(text, str) or text.strip() == "":
            continue
        for token in extract_ngrams(text, n=NGRAM_MAX_N):
            all_words.append(token)

    freq = Counter(all_words)
    candidates = [(w, c) for w, c in freq.items() if c > MIN_TERM_FREQ]
    candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
    candidates = candidates[:MAX_CANDIDATES]

    print(f"Kandidaten nach Mindesthäufigkeit: {len(candidates):,}")
    print(f"Maximale Kandidatenzahl: {MAX_CANDIDATES:,}")
    return [w for w, _ in candidates]

# 4: BERT-Embedding für Wörter
def embed_words(words):
    print(f"Erzeuge Embeddings für {len(words):,} Kandidaten...")
    vectors = bert.encode(words, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=True)
    return np.array(vectors), words

# 5: Topic-Seed-Vektoren vorbereiten
print("Erzeuge Embeddings für Topic-Seeds...")
TOPIC_VECS = {topic: bert.encode(seeds, batch_size=EMBEDDING_BATCH_SIZE) for topic, seeds in TOPIC_SEEDS.items()}

# 6: Topics zuweisen (ohne Sentiment)
def assign_topics_with_confidence(words, word_vectors):
    print("Weise Topics zu...")
    topic_rows = []
    for w, v in zip(words, word_vectors):
        sims = {}
        for topic, seed_vecs in TOPIC_VECS.items():
            sims[topic] = np.mean([
                np.dot(v, sv) / (np.linalg.norm(v) * np.linalg.norm(sv))
                for sv in seed_vecs
            ])
        sim_values = list(sims.values())
        min_sim, max_sim = min(sim_values), max(sim_values)
        span = max_sim - min_sim if max_sim != min_sim else 1.0

        for topic, sim in sims.items():
            conf = (sim - min_sim) / span  # 0–1 normalisiert
            if conf > MIN_TOPIC_CONFIDENCE:  # nur starke Zuordnung
                topic_rows.append({
                    "text": w,
                    "name_cluster": topic,
                    "confidence": conf
                })
    print(f"Topic-Lexikon-Zeilen: {len(topic_rows):,}")
    return topic_rows

# 7: Lexikon bauen
def build_topic_lexicon(df):
    words = build_candidate_words(df)

    if not words:
        raise ValueError("Keine Kandidaten für das Topic-Lexikon gefunden.")

    word_vectors, words = embed_words(words)
    topic_rows = assign_topics_with_confidence(words, word_vectors)
    return pd.DataFrame(topic_rows)

# 8: Main
print("Lade Yelp-Daten...")
df = pd.read_csv(INPUT_FILE)
print(f"Geladene Zeilen: {len(df):,}")

if TEXT_COLUMN not in df.columns:
    raise ValueError(f"Spalte fehlt in yelp_final.csv: {TEXT_COLUMN}")

df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
df = df[df[TEXT_COLUMN].str.strip().str.lower() != "nan"]
df = df[df[TEXT_COLUMN].str.strip() != ""]

if len(df) > MAX_ROWS:
    print(f"Ziehe zufällige Stichprobe mit {MAX_ROWS:,} von {len(df):,} Zeilen.")
    df = df.sample(n=MAX_ROWS, random_state=RANDOM_STATE).copy()

print(f"Verwendete Zeilen: {len(df):,}")

lexicon_df = build_topic_lexicon(df)
lexicon_df.to_excel(OUTPUT_FILE, index=False)

print("Fertig! Themenlexikon (Seeds) erstellt.")
print(f"Output-Datei: {OUTPUT_FILE}")





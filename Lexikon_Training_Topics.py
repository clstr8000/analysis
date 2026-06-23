# Lexikon erstellen: Themenlexikon ohne Sentiment, mit festen Seeds und Review-Kandidaten

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


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
SEED_CONFIDENCE = 1.0

TOPIC_SEEDS = {
    "FOOD": [
        "food", "burger", "sandwich", "chicken", "fries", "nuggets", "meal",
        "taste", "flavor", "sauce", "portion", "fresh", "delicious", "tasty",
        "crispy", "juicy", "spicy", "bland", "soggy", "cold food", "hot food",
        "overcooked", "undercooked", "burnt", "salty", "seasoning", "texture",
        "quality", "freshness", "temperature", "lukewarm", "dry chicken",
        "tender chicken", "filet", "chicken filet", "original filet", "spicy filet",
        "grilled filet", "pickle", "pickles", "bun", "multigrain bun",
        "chick-fil-a chicken biscuit", "chick-fil-a chicken biscuit meal",
        "spicy chicken biscuit", "spicy chicken biscuit meal",
        "chick-fil-a chick-n-minis", "chick-n-minis", "chick-n-minis meal",
        "4 ct chick-n-minis", "10 ct chick-n-minis", "egg white grill",
        "egg white grill meal", "hash brown scramble burrito",
        "hash brown scramble burrito meal", "hash brown scramble bowl",
        "hash brown scramble bowl meal", "chicken egg cheese biscuit",
        "bacon egg cheese biscuit", "sausage egg cheese biscuit",
        "chicken egg cheese muffin", "bacon egg cheese muffin",
        "sausage egg cheese muffin", "english muffin", "biscuit", "hash browns",
        "jalapeno salsa", "berry parfait", "greek yogurt parfait", "fruit cup",
        "breakfast filet", "breakfast burrito", "breakfast bowl",
        "chick-fil-a chicken sandwich", "chick-fil-a chicken sandwich meal",
        "chick-fil-a deluxe sandwich", "chick-fil-a deluxe sandwich meal",
        "spicy chicken sandwich", "spicy chicken sandwich meal",
        "spicy deluxe sandwich", "spicy deluxe sandwich meal",
        "grilled chicken sandwich", "grilled chicken sandwich meal",
        "grilled chicken club sandwich", "grilled chicken club sandwich meal",
        "grilled spicy deluxe sandwich", "grilled spicy deluxe sandwich meal",
        "honey pepper pimento sandwich", "honey pepper pimento sandwich meal",
        "honey pepper pimento", "pimento cheese", "pickled jalapenos",
        "jalapeno ranch club sandwich", "jalapeno ranch club sandwich meal",
        "jalapeño ranch club sandwich", "jalapeño ranch club sandwich meal",
        "jalapeno ranch", "jalapeño ranch",
        "chick-fil-a nuggets", "chick-fil-a nuggets meal", "nuggets meal",
        "5 ct nuggets", "8 ct nuggets", "12 ct nuggets", "30 ct nuggets",
        "8 ct chick-fil-a nuggets", "12 ct chick-fil-a nuggets",
        "30 ct chick-fil-a nuggets", "grilled nuggets", "grilled nuggets meal",
        "5 ct grilled nuggets", "8 ct grilled nuggets", "12 ct grilled nuggets",
        "30 ct grilled nuggets", "chick-n-strips", "chicken strips",
        "chick-n-strips meal", "2 ct chick-n-strips", "3 ct chick-n-strips",
        "4 ct chick-n-strips", "cool wrap", "chick-fil-a cool wrap",
        "cool wrap meal", "wrap", "grilled cool wrap",
        "cobb salad", "spicy southwest salad", "market salad", "side salad",
        "salad", "salads", "roasted corn", "black beans", "tortilla strips",
        "pepitas", "blue cheese", "granola", "harvest nut granola", "roasted almonds",
        "waffle potato fries", "waffle fries", "fries", "mac and cheese",
        "mac & cheese", "chicken noodle soup", "soup", "waffle potato chips",
        "original flavor waffle potato chips", "chick-fil-a sauce flavored waffle potato chips",
        "apple sauce", "buddy fruits apple sauce", "kale crunch side", "kale salad",
        "kale crunch", "side item", "side items",
        "5 ct nuggets kid's meal", "5 ct nuggets kids meal", "nuggets kid's meal",
        "nuggets kids meal", "2 ct chick-n-strips kid's meal",
        "2 ct chick-n-strips kids meal", "chick-n-strips kid's meal",
        "chick-n-strips kids meal", "grilled nuggets kid's meal",
        "grilled nuggets kids meal", "mac & cheese kid's meal", "mac and cheese kid's meal",
        "kids meal", "kid's meal", "family style meal", "family meal",
        "family style entree", "family style side", "family style treat",
        "chick-fil-a meal", "entree", "entrees", "packaged meal", "boxed meal",
        "catering", "catering menu", "nugget tray", "chick-fil-a nugget tray",
        "chilled nugget tray", "grilled nugget tray", "chick-n-strips tray",
        "cool wrap tray", "fruit tray", "garden salad tray", "spicy southwest salad tray",
        "cobb salad tray", "market salad tray", "mac and cheese tray", "mac & cheese tray",
        "cookie tray", "brownie tray", "catering tray", "peach milkshake",
        "vanilla milkshake", "cookies cream milkshake", "cookies & cream milkshake",
        "chocolate milkshake", "strawberry milkshake", "icedream cup", "icedream cone",
        "icedream", "chocolate chunk cookie", "chocolate fudge brownie", "brownie",
        "cookie", "treat", "treats", "barbeque sauce", "bbq sauce",
        "chick-fil-a sauce", "garden herb ranch sauce", "honey mustard sauce",
        "polynesian sauce", "sweet spicy sriracha sauce", "sweet & spicy sriracha sauce",
        "zesty buffalo sauce", "honey roasted bbq sauce", "avocado lime ranch dressing",
        "creamy salsa dressing", "fat-free honey mustard dressing", "fat free honey mustard dressing",
        "garden herb ranch dressing", "light balsamic vinaigrette dressing",
        "light italian dressing", "zesty apple cider vinaigrette dressing", "dressing",
        "dressings", "sauces", "dipping sauce", "dipping sauces"
    ],
    "DRINKS": [
        "drink", "drinks", "beverage", "beverages", "soda", "tea", "sweet tea",
        "iced tea", "lemonade", "water", "coffee", "milkshake", "shake", "refill",
        "fountain drink", "fountain beverage", "ice", "watery", "cold drink", "hot drink",
        "carbonated", "flat soda", "fresh lemonade", "pineapple dragonfruit sprite",
        "pineapple dragonfruit & sprite", "pineapple dragonfruit lemonade",
        "pineapple dragonfruit sunjoy", "pineapple dragonfruit teas",
        "pineapple dragonfruit beverage", "chick-fil-a lemonade", "diet lemonade",
        "sunjoy", "half sweet tea half lemonade", "half sweet tea half diet lemonade",
        "half unsweet tea half lemonade", "half unsweet tea half diet lemonade",
        "freshly-brewed sweetened iced tea", "freshly-brewed unsweetened iced tea",
        "freshly brewed sweetened iced tea", "freshly brewed unsweetened iced tea",
        "sweetened iced tea", "unsweetened iced tea", "iced coffee", "hot coffee",
        "dasani bottled water", "bottled water", "honest kids apple juice", "apple juice",
        "simply orange", "orange juice", "chocolate milk", "1% chocolate milk",
        "milk", "1% milk", "dr pepper", "coca-cola", "coke", "sprite", "diet coke",
        "coke zero", "root beer", "fanta", "hi-c", "powerade", "seasonal gallon beverages",
        "gallon beverages", "gallon sweet tea", "gallon unsweet tea", "gallon lemonade",
        "gallon diet lemonade", "catering beverage", "beverage gallon",
        "pineapple dragonfruit frosted lemonade", "peach frosted lemonade",
        "frosted lemonade", "frosted sodas", "frosted soda", "floats", "float",
        "frosted coffee", "icedream float"
    ],
    "SERVICE": ["service", "staff", "employee", "employees", "worker", "workers", "cashier", "manager", "friendly", "helpful", "polite", "rude", "attentive", "unprofessional", "customer service", "greeted", "welcoming", "attitude", "respectful", "disrespectful", "kind", "courteous", "patient", "impatient", "professional", "pleasant", "smiling", "smile", "care", "caring", "hospitality", "team member", "crew", "server", "assistance", "help", "ignored", "apologized", "accommodating", "knowledgeable", "training"],
    "SPEED": ["fast", "slow", "quick", "quickly", "speed", "speedy", "service speed", "order ready", "ready fast", "took forever", "took too long", "delay", "delayed", "immediate", "efficient", "inefficient", "wait", "waiting", "wait time", "long wait", "short wait", "line", "queue", "long line", "short line", "stood in line", "standing in line", "crowded line", "busy", "rush", "lunch rush", "dinner rush", "peak hour", "served quickly", "served fast", "ready quickly", "slow service", "fast service", "order time", "pickup time", "turnaround", "backed up"],
    "HYGIENE": ["clean", "dirty", "messy", "filthy", "spotless", "sanitary", "unsanitary", "hygiene", "cleanliness", "sticky", "smell", "odor", "trash", "garbage", "bathroom", "restroom", "table", "tables", "floor", "floors", "counter", "counters", "sink", "toilet", "napkin", "spill", "spilled", "greasy", "dusty", "stain", "stained", "cleaned", "unclean", "neat", "tidy", "sanitized", "soap", "paper towel", "overflowing trash"],
    "VALUE": ["price", "prices", "expensive", "cheap", "value", "worth", "deal", "combo", "meal deal", "affordable", "overpriced", "cost", "charged", "fair price", "reasonable", "unreasonable", "portion size", "money", "receipt", "bill", "total", "discount", "coupon", "reward", "rewards", "points", "free item", "promotion", "special", "pricey", "low price", "high price", "good value", "bad value", "not worth", "worth it", "small portion", "large portion"],
    "AMBIENCE": ["ambience", "atmosphere", "vibe", "environment", "seating", "dining room", "inside", "restaurant", "location", "crowded", "quiet", "loud", "noisy", "comfortable", "uncomfortable", "decor", "music", "lighting", "space", "layout", "interior", "booth", "chair", "chairs", "table", "tables", "family friendly", "kid friendly", "play area", "temperature", "air conditioning", "warm", "cold", "busy atmosphere", "calm", "relaxing", "modern"],
    "PARKING": ["parking", "parking lot", "parking spot", "parking spaces", "car", "cars", "lot", "garage", "parked", "hard to park", "easy parking", "traffic", "entrance", "exit", "street parking", "curbside", "curb", "pickup spot", "parking area", "small lot", "full lot", "crowded lot", "traffic flow", "turn in", "turn out", "blocked", "congested", "nearby parking", "free parking"],
    "ACCESSIBILITY": ["wheelchair", "accessible", "accessibility", "entrance", "ramp", "stairs", "disabled", "handicap", "elevator", "restroom access", "parking access", "easy access", "door", "automatic door", "wide door", "narrow door", "mobility", "walker", "stroller", "step", "curb", "sidewalk", "path", "space", "accessible parking", "handicap parking", "accessible table", "accessible restroom", "low counter", "high counter"],
    "GENERAL": ["good", "bad", "great", "nice", "amazing", "terrible", "awful", "excellent", "fine", "okay", "average", "experience", "visit", "place", "spot", "location", "recommend", "disappointed", "satisfied", "overall", "favorite", "best", "worst", "love", "liked", "enjoyed", "happy", "unhappy", "return", "come back", "never again", "consistent", "inconsistent"]
}

print("Lade Embedding-Modell...")
bert = SentenceTransformer("all-MiniLM-L6-v2")


def extract_ngrams(text, n=3):
    tokens = [t.lower() for t in text.split() if t.replace("-", "").isalpha()]
    ngrams = []
    for t in tokens:
        if len(t) > 2:
            ngrams.append(t)
    for size in range(2, n + 1):
        for i in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[i:i + size])
            if all(len(w) > 2 for w in tokens[i:i + size]):
                ngrams.append(phrase)
    return ngrams


def build_candidate_words(df):
    print("Extrahiere Kandidaten aus Review-Texten...")
    all_words = []
    for text in df[TEXT_COLUMN]:
        if isinstance(text, str) and text.strip():
            all_words.extend(extract_ngrams(text, n=NGRAM_MAX_N))
    freq = Counter(all_words)
    candidates = [(w, c) for w, c in freq.items() if c > MIN_TERM_FREQ]
    candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[:MAX_CANDIDATES]
    print(f"Kandidaten nach Mindesthäufigkeit: {len(candidates):,}")
    return [w for w, _ in candidates]


def embed_words(words):
    print(f"Erzeuge Embeddings für {len(words):,} Kandidaten...")
    vectors = bert.encode(words, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=True)
    return np.array(vectors), words


print("Erzeuge Embeddings für Topic-Seeds...")
TOPIC_VECS = {topic: bert.encode(seeds, batch_size=EMBEDDING_BATCH_SIZE) for topic, seeds in TOPIC_SEEDS.items()}


def assign_topics_with_confidence(words, word_vectors):
    print("Weise Topics zu...")
    topic_rows = []
    for w, v in zip(words, word_vectors):
        sims = {}
        for topic, seed_vecs in TOPIC_VECS.items():
            sims[topic] = np.mean([np.dot(v, sv) / (np.linalg.norm(v) * np.linalg.norm(sv)) for sv in seed_vecs])
        sim_values = list(sims.values())
        min_sim, max_sim = min(sim_values), max(sim_values)
        span = max_sim - min_sim if max_sim != min_sim else 1.0
        for topic, sim in sims.items():
            conf = (sim - min_sim) / span
            if conf > MIN_TOPIC_CONFIDENCE:
                topic_rows.append({"text": w, "name_cluster": topic, "confidence": conf, "source": "candidate_similarity"})
    print(f"Topic-Lexikon-Zeilen aus Kandidaten: {len(topic_rows):,}")
    return topic_rows


def add_seed_rows(lexicon_df):
    print("Ergänze Topic-Seeds im Lexikon...")
    seed_rows = []
    for topic, seeds in TOPIC_SEEDS.items():
        for seed in seeds:
            seed_rows.append({"text": str(seed).lower().strip(), "name_cluster": topic, "confidence": SEED_CONFIDENCE, "source": "topic_seed"})
    seed_df = pd.DataFrame(seed_rows)
    combined_df = pd.concat([lexicon_df, seed_df], ignore_index=True)
    combined_df = combined_df.sort_values(["text", "confidence"], ascending=[True, False])
    combined_df = combined_df.drop_duplicates(subset=["text"], keep="first")
    combined_df = combined_df.sort_values(["name_cluster", "text"]).reset_index(drop=True)
    print(f"Topic-Seeds ergänzt: {len(seed_df):,}")
    print(f"Topic-Lexikon-Zeilen gesamt: {len(combined_df):,}")
    return combined_df


def build_topic_lexicon(df):
    words = build_candidate_words(df)
    if not words:
        raise ValueError("Keine Kandidaten für das Topic-Lexikon gefunden.")
    word_vectors, words = embed_words(words)
    topic_rows = assign_topics_with_confidence(words, word_vectors)
    return add_seed_rows(pd.DataFrame(topic_rows))


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

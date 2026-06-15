# Lexikon erstellen (Version 6 – Themenlexikon ohne Sentiment, mit Cluster-Namen und 1–3-Wort-Ausdrücken)

# 0.0: Packages importieren
import pandas as pd  # (pandas für Tabellen, Datenverwendung)
import numpy as np   # (numpy auch für Tabellen, Datenverwendung)
from collections import Counter  # (counter um Worthäufigkeiten zu zählen)
from sentence_transformers import SentenceTransformer  # (enthält BERT)
from pathlib import Path

# 0.1: BERT laden
bert = SentenceTransformer("all-MiniLM-L6-v2")  # (BERT analysiert Text kontextuell)

# 1: Themenwörter zur Orientierung (Seeds)
TOPIC_SEEDS = {
    "FOOD": [
        "food", "burger", "sandwich", "chicken", "fries", "nuggets",
        "meal", "taste", "flavor", "sauce", "portion", "fresh",
        "delicious", "tasty", "crispy", "juicy", "spicy", "bland",
        "soggy", "cold food", "hot food", "overcooked", "undercooked",
        "burnt", "salty", "seasoning"
    ],

    "SERVICE": [
        "service", "staff", "employee", "employees", "worker", "workers",
        "cashier", "manager", "friendly", "helpful", "polite", "rude",
        "attentive", "unprofessional", "customer service", "greeted",
        "welcoming", "attitude", "respectful", "disrespectful"
    ],

    "ORDER_ACCURACY": [
        "wrong order", "incorrect order", "missing item", "missing items",
        "forgot", "forgotten", "left out", "messed up order",
        "order was wrong", "gave me wrong", "did not receive",
        "missing sauce", "no sauce", "wrong sauce", "wrong drink",
        "wrong sandwich", "wrong meal", "order accuracy", "accurate order"
    ],

    "SPEED": [
        "fast", "slow", "quick", "quickly", "speed", "speedy",
        "service speed", "order ready", "ready fast", "took forever",
        "took too long", "delay", "delayed", "immediate", "efficient",
        "inefficient"
    ],

    "WAITING": [
        "wait", "waiting", "wait time", "long wait", "short wait",
        "line", "queue", "long line", "short line", "stood in line",
        "standing in line", "crowded line", "busy", "rush", "lunch rush"
    ],

    "DRIVE_THRU": [
        "drive thru", "drive-thru", "drive through", "drive-through",
        "drive thru line", "drive-thru line", "lane", "lanes",
        "speaker", "menu board", "window", "pickup window",
        "drive thru service", "drive thru order", "car line"
    ],

    "HYGIENE": [
        "clean", "dirty", "messy", "filthy", "spotless", "sanitary",
        "unsanitary", "hygiene", "cleanliness", "sticky", "smell",
        "odor", "trash", "garbage", "bathroom", "restroom",
        "table", "tables", "floor", "floors"
    ],

    "VALUE": [
        "price", "prices", "expensive", "cheap", "value", "worth",
        "deal", "combo", "meal deal", "affordable", "overpriced",
        "cost", "charged", "fair price", "reasonable", "unreasonable",
        "portion size"
    ],

    "AMBIENCE": [
        "ambience", "atmosphere", "vibe", "environment", "seating",
        "dining room", "inside", "restaurant", "location", "crowded",
        "quiet", "loud", "noisy", "comfortable", "uncomfortable",
        "decor", "music", "lighting"
    ],

    "PARKING": [
        "parking", "parking lot", "parking spot", "parking spaces",
        "car", "cars", "lot", "garage", "parked", "hard to park",
        "easy parking", "traffic", "entrance", "exit"
    ],

    "DRINKS": [
        "drink", "drinks", "beverage", "soda", "tea", "sweet tea",
        "iced tea", "lemonade", "water", "coffee", "milkshake",
        "shake", "refill", "fountain drink", "ice", "watery"
    ],

    "ACCESSIBILITY": [
        "wheelchair", "accessible", "accessibility", "entrance",
        "ramp", "stairs", "disabled", "handicap", "elevator",
        "restroom access", "parking access", "easy access"
    ],

    "GENERAL": [
        "good", "bad", "great", "nice", "amazing", "terrible",
        "awful", "excellent", "fine", "okay", "average",
        "experience", "visit", "place", "spot", "location",
        "recommend", "disappointed", "satisfied"
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
    all_words = []
    for text in df["text"]:
        if not isinstance(text, str) or text.strip() == "":
            continue
        for token in extract_ngrams(text, n=3):
            all_words.append(token)
    freq = Counter(all_words)
    return [w for w, c in freq.items() if c > 5]  # Wörter müssen >5 vorkommen

# 4: BERT-Embedding für Wörter
def embed_words(words):
    vectors = bert.encode(words, batch_size=64, show_progress_bar=True)
    return np.array(vectors), words

# 5: Topic-Seed-Vektoren vorbereiten
TOPIC_VECS = {topic: bert.encode(seeds) for topic, seeds in TOPIC_SEEDS.items()}

# 6: Topics zuweisen (ohne Sentiment)
def assign_topics_with_confidence(words, word_vectors):
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
            if conf > 0.7:  # nur starke Zuordnung
                topic_rows.append({
                    "text": w,
                    "name_cluster": topic,
                    "confidence": conf
                })
    return topic_rows

# 7: Lexikon bauen
def build_topic_lexicon(df):
    words = build_candidate_words(df)
    word_vectors, words = embed_words(words)
    topic_rows = assign_topics_with_confidence(words, word_vectors)
    return pd.DataFrame(topic_rows)

# 8: Main
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INPUT_FILE = PROJECT_DIR / "yelp_final.csv"
OUTPUT_FILE = BASE_DIR / "topic_lexicon_seeds_only.xlsx"

df = pd.read_csv(INPUT_FILE)
df["text"] = df["text"].astype(str)
df = df[df["text"].str.strip().str.lower() != "nan"]
df = df[df["text"].str.strip() != ""]

lexicon_df = build_topic_lexicon(df)
lexicon_df.to_excel(OUTPUT_FILE, index=False)

print("Fertig! Themenlexikon (Seeds) erstellt.")





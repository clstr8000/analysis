# Lexikon erstellen: Themenlexikon ohne Sentiment, mit festen Seeds und Review-Kandidaten

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from chick_fil_a_topic_seeds import TOPIC_SEEDS


# =========================================================
# KONFIGURATION
# =========================================================
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


# =========================================================
# MODELL
# =========================================================
print("Lade Embedding-Modell...")
bert = SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# HILFSFUNKTIONEN
# =========================================================
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
        if not isinstance(text, str) or text.strip() == "":
            continue
        all_words.extend(extract_ngrams(text, n=NGRAM_MAX_N))

    freq = Counter(all_words)
    candidates = [(w, c) for w, c in freq.items() if c > MIN_TERM_FREQ]
    candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
    candidates = candidates[:MAX_CANDIDATES]

    print(f"Kandidaten nach Mindesthäufigkeit: {len(candidates):,}")
    print(f"Maximale Kandidatenzahl: {MAX_CANDIDATES:,}")
    return [w for w, _ in candidates]


def embed_words(words):
    print(f"Erzeuge Embeddings für {len(words):,} Kandidaten...")
    vectors = bert.encode(words, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=True)
    return np.array(vectors), words


print("Erzeuge Embeddings für Topic-Seeds...")
TOPIC_VECS = {
    topic: bert.encode(seeds, batch_size=EMBEDDING_BATCH_SIZE)
    for topic, seeds in TOPIC_SEEDS.items()
}


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
            conf = (sim - min_sim) / span
            if conf > MIN_TOPIC_CONFIDENCE:
                topic_rows.append({
                    "text": w,
                    "name_cluster": topic,
                    "confidence": conf,
                    "source": "candidate_similarity"
                })

    print(f"Topic-Lexikon-Zeilen aus Kandidaten: {len(topic_rows):,}")
    return topic_rows


def add_seed_rows(lexicon_df):
    print("Ergänze Topic-Seeds im Lexikon...")
    seed_rows = []

    for topic, seeds in TOPIC_SEEDS.items():
        for seed in seeds:
            seed_rows.append({
                "text": str(seed).lower().strip(),
                "name_cluster": topic,
                "confidence": SEED_CONFIDENCE,
                "source": "topic_seed"
            })

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
    lexicon_df = pd.DataFrame(topic_rows)
    return add_seed_rows(lexicon_df)


# =========================================================
# MAIN
# =========================================================
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

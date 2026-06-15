# Lexikon erstellen (Version 6 – Automatische Cluster nur für Chick-fil-A)

# 0.0: Packages importieren
import pandas as pd
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import re


# 0.1: BERT laden
bert = SentenceTransformer("all-MiniLM-L6-v2")

# Chick-fil-A Schreibweisen
CFA_PATTERNS = [
    r"chick[\s\-]?fil[\s\-]?a",
    r"chickfila",
    r"chik[\s\-]?fil[\s\-]?a",
    r"cfa\b"
]
CFA_REGEX = re.compile("|".join(CFA_PATTERNS), flags=re.IGNORECASE)

# 1: N-Gramme extrahieren (1–3 Wörter)
def extract_ngrams(text, n=3):
    tokens = [t.lower() for t in text.split() if t.replace("-", "").isalpha()]
    ngrams = []

    # Unigramme
    for t in tokens:
        if len(t) > 2:
            ngrams.append(t)

    # Bi- und Trigramme
    for size in range(2, n+1):
        for i in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[i:i+size])
            if all(len(w) > 2 for w in tokens[i:i+size]):
                ngrams.append(phrase)

    return ngrams

# 2: Wörter extrahieren
def build_candidate_words(df):
    all_words = []
    for text in df["text"]:
        if not isinstance(text, str) or text.strip() == "":
            continue
        for token in extract_ngrams(text, n=3):
            all_words.append(token)
    freq = Counter(all_words)
    return [w for w, c in freq.items() if c > 5]

# 3: Embeddings
def embed_words(words):
    vectors = bert.encode(words, batch_size=64, show_progress_bar=True)
    return np.array(vectors), words

# 4: Clustering
def cluster_words(words, word_vectors, n_clusters=30):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(word_vectors)
    centers = kmeans.cluster_centers_

    rows = []
    for w, v, label in zip(words, word_vectors, labels):
        center = centers[label]
        sim = cosine_similarity([v], [center])[0][0]
        if sim > 0.7:
            rows.append({
                "lemma": w,
                "cluster_id": int(label),
                "confidence": float(sim)
            })
    return pd.DataFrame(rows)

# 5: Cluster benennen
def name_clusters(df):
    names = {}
    for cid in df["cluster_id"].unique():
        subset = df[df["cluster_id"] == cid].sort_values("confidence", ascending=False)
        top_words = subset["lemma"].head(3).tolist()
        name = "TOPIC_" + "_".join(top_words)
        names[cid] = name

    df["name_cluster"] = df["cluster_id"].map(names)
    df = df[["lemma", "name_cluster", "confidence"]]
    df = df.rename(columns={"lemma": "text"})
    return df

# 6: Main
df = pd.read_csv("../yelp_final.csv")
df["text"] = df["text"].astype(str)

# Nur echte Texte behalten
df = df[df["text"].str.strip().str.lower() != "nan"]
df = df[df["text"].str.strip() != ""]

# ⭐ NUR Chick-fil-A Reviews filtern
df["is_cfa"] = df["business_name"].str.contains(CFA_REGEX, na=False)
df_cfa = df[df["is_cfa"]].copy()

print(f"Gefundene Chick-fil-A Reviews: {len(df_cfa)}")

# Wörter nur aus Chick-fil-A Reviews
words = build_candidate_words(df_cfa)
word_vectors, words = embed_words(words)

cluster_df = cluster_words(words, word_vectors, n_clusters=30)
cluster_df = name_clusters(cluster_df)

cluster_df.to_excel("topic_clusters_auto_cfa_only.xlsx", index=False)

print("Fertig! Automatische Chick-fil-A-Cluster erstellt.")


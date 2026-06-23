# ============================================================
# ABSA-Sentiment-Pipeline mit Rating-Lexikon, Topic-Lexikon,
# Topic-Seed-Fallback, Satzanalyse und Review-Aggregation
# ============================================================

from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re

import numpy as np
import pandas as pd
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from chick_fil_a_topic_seeds import TOPIC_SEEDS


# ============================================================
# 0. Pfade, Spalten und Modelle
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

REVIEWS_FILE = DATA_DIR / "yelp_final.csv"
RATING_LEXICON_FILE = BASE_DIR / "generated_sentiment_lexicon_latest.xlsx"
TOPIC_LEXICON_FILE = BASE_DIR / "topic_lexicon_seeds_only.xlsx"
OUTPUT_SENTENCE_LEVEL = BASE_DIR / "absa_sentence_level.xlsx"
OUTPUT_REVIEW_LEVEL = BASE_DIR / "absa_review_level.xlsx"

TEXT_COLUMN = "text"
BUSINESS_NAME_COLUMN = "business_name"
PROGRESS_EVERY = 500

CFA_PATTERNS = [
    r"chick[\s\-]?fil[\s\-]?a",
    r"chickfila",
    r"chik[\s\-]?fil[\s\-]?a",
    r"cfa\b"
]
CFA_REGEX = re.compile("|".join(CFA_PATTERNS), flags=re.IGNORECASE)

TOPIC_WORDS = {
    topic: {str(seed).lower().strip() for seed in seeds}
    for topic, seeds in TOPIC_SEEDS.items()
}

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_000_000

SENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
sent_tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL)
sent_model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL)


# ============================================================
# 1. Input und Lexika laden
# ============================================================
def load_cfa_reviews(path):
    print("Lade Yelp-Daten...")
    df = pd.read_csv(path)
    print(f"Geladene Zeilen: {len(df):,}")

    missing_columns = {TEXT_COLUMN, BUSINESS_NAME_COLUMN} - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Diese Spalten fehlen in yelp_final.csv: {sorted(missing_columns)}"
        )

    df = df[df[TEXT_COLUMN].notna()].copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    df = df[df[TEXT_COLUMN].str.strip() != ""]
    df = df[df[TEXT_COLUMN].str.strip().str.lower() != "nan"]

    is_cfa = df[BUSINESS_NAME_COLUMN].astype(str).str.contains(CFA_REGEX, na=False)
    df_cfa = df[is_cfa].copy()
    print(f"Gefundene Chick-fil-A-Bewertungen: {len(df_cfa):,}")

    if df_cfa.empty:
        raise ValueError("Keine Chick-fil-A-Bewertungen in yelp_final.csv gefunden.")

    return df_cfa


def row_value(row, columns, default=None):
    for column in columns:
        if column in row and pd.notna(row[column]):
            return row[column]
    return default


def load_rating_lexicon(path):
    print("Lade Rating-Lexikon...")
    df = pd.read_excel(path)
    lex = {}
    for _, row in df.iterrows():
        term = str(row["text"]).lower().strip()
        rating = int(row["rating"])
        conf = float(row["confidence"])
        lex[term] = (rating, conf)
    print(f"Rating-Lexikon-Terme: {len(lex):,}")
    return lex


def load_topic_lexicon(path):
    print("Lade Topic-Lexikon...")
    df = pd.read_excel(path)
    lex = {}
    for _, row in df.iterrows():
        term = str(row["text"]).lower().strip()
        topic = str(row["name_cluster"])
        conf = float(row["confidence"])
        lex[term] = (topic, conf)
    print(f"Topic-Lexikon-Terme: {len(lex):,}")
    return lex


# ============================================================
# 2. Sentiment und Topics
# ============================================================
def bert_sentence_rating(sent):
    inputs = sent_tokenizer(sent.text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = sent_model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]
    rating = int(np.argmax(probs) + 1)
    confidence = float(probs.max())
    return rating, confidence


def combine_rating(term, sent, rating_lex):
    term_norm = term.lower().strip()
    bert_rating, bert_conf = bert_sentence_rating(sent)

    if term_norm in rating_lex:
        lex_rating, lex_conf = rating_lex[term_norm]
        alpha = max(0.0, min(1.0, lex_conf))
        final_rating = alpha * lex_rating + (1 - alpha) * bert_rating
        final_conf = alpha * lex_conf + (1 - alpha) * bert_conf
        return final_rating, final_conf, "lexicon+bert"

    return bert_rating, bert_conf, "bert_only"


def get_topic(term, topic_lex):
    term_norm = term.lower().strip()

    if term_norm in topic_lex:
        topic, conf = topic_lex[term_norm]
        return topic, conf

    for topic, words in TOPIC_WORDS.items():
        if term_norm in words:
            return topic, 1.0

    return "GENERAL", 0.0


# ============================================================
# 3. Aspekte und Satzanalyse
# ============================================================
def extract_aspects(sent):
    aspects = []
    for token in sent:
        if token.pos_ == "NOUN":
            base = token.lemma_.lower()
            compounds = [child.text.lower() for child in token.children if child.dep_ == "compound"]
            if compounds:
                aspects.append(" ".join(compounds + [token.text.lower()]))
            else:
                aspects.append(base)
    return list(set(aspects))


def analyze_sentence(sent, rating_lex, topic_lex):
    aspects = extract_aspects(sent)
    results = []

    if not aspects:
        rating, conf, src = combine_rating(sent.text, sent, rating_lex)
        results.append({
            "sentence": sent.text,
            "text": sent.text,
            "rating": rating,
            "confidence": conf,
            "topic": "GENERAL",
            "topic_confidence": 0.0,
            "source": src
        })
        return results

    for asp in aspects:
        rating, conf, src = combine_rating(asp, sent, rating_lex)
        topic, topic_conf = get_topic(asp, topic_lex)
        results.append({
            "sentence": sent.text,
            "text": asp,
            "rating": rating,
            "confidence": conf,
            "topic": topic,
            "topic_confidence": topic_conf,
            "source": src
        })

    return results


# ============================================================
# 4. Review-Pipeline
# ============================================================
def run_absa_pipeline(df_reviews, rating_lex, topic_lex):
    sentence_rows = []
    review_rows = []
    total_reviews = len(df_reviews)

    print(f"Starte ABSA für {total_reviews:,} Reviews...")

    for idx, (_, row) in enumerate(df_reviews.iterrows(), start=1):
        if idx == 1 or idx % PROGRESS_EVERY == 0 or idx == total_reviews:
            print(f"Verarbeite Review {idx:,}/{total_reviews:,}...")

        text = row[TEXT_COLUMN]
        user_id = row_value(row, ["user_id", "author_id", "reviewer_id"])
        store_id = row_value(row, ["store_id", "business_id", "location_id", BUSINESS_NAME_COLUMN])
        time_raw = row_value(row, ["time", "date", "created_at", "review_date"])

        try:
            dt = datetime.strptime(str(time_raw), "%d.%m.%Y %H:%M:%S")
            time_str = dt.isoformat()
        except Exception:
            time_str = str(time_raw)

        doc = nlp(text)
        all_aspects = []

        for sent in doc.sents:
            sent_results = analyze_sentence(sent.as_doc(), rating_lex, topic_lex)
            for r in sent_results:
                r["user_id"] = user_id
                r["store_id"] = store_id
                r["time"] = time_str
                sentence_rows.append(r)
                all_aspects.append(r)

        topic_groups = defaultdict(list)
        for asp in all_aspects:
            topic_groups[asp["topic"]].append((asp["rating"], asp["confidence"]))

        review_result = {
            "user_id": user_id,
            "store_id": store_id,
            "time": time_str
        }

        for topic in list(TOPIC_WORDS.keys()):
            vals = topic_groups.get(topic, [])
            if not vals:
                review_result[topic] = None
            else:
                ratings = np.array([v[0] for v in vals])
                confs = np.array([v[1] for v in vals])
                review_result[topic] = np.sum(ratings * confs) / np.sum(confs)

        review_rows.append(review_result)

    print(f"Satz-Level-Zeilen: {len(sentence_rows):,}")
    print(f"Review-Level-Zeilen: {len(review_rows):,}")
    return pd.DataFrame(sentence_rows), pd.DataFrame(review_rows)


# ============================================================
# 5. MAIN
# ============================================================
if __name__ == "__main__":
    df_reviews = load_cfa_reviews(REVIEWS_FILE)
    rating_lex = load_rating_lexicon(RATING_LEXICON_FILE)
    topic_lex = load_topic_lexicon(TOPIC_LEXICON_FILE)

    df_sentences, df_reviews_agg = run_absa_pipeline(df_reviews, rating_lex, topic_lex)

    print("Speichere ABSA-Outputs...")
    df_sentences.to_excel(OUTPUT_SENTENCE_LEVEL, index=False)
    df_reviews_agg.to_excel(OUTPUT_REVIEW_LEVEL, index=False)

    print("ABSA Export abgeschlossen.")
    print(f"Verwendete Chick-fil-A-Bewertungen: {len(df_reviews):,}")
    print(f"Satz-Level-Datei: {OUTPUT_SENTENCE_LEVEL}")
    print(f"Review-Level-Datei: {OUTPUT_REVIEW_LEVEL}")

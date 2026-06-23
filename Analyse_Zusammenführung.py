# ============================================================
# ABSA-Sentiment-Pipeline (mit echtem BERT, Rating-Lexikon,
# Topic-Lexikon, Satzanalyse und Review-Aggregation)
# ============================================================

from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ------------------------------------------------------------
# 0. Pfade und Modelle laden
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

REVIEWS_FILE = DATA_DIR / "ChickFilA_Bereinigt.xlsx"
RATING_LEXICON_FILE = BASE_DIR / "generated_sentiment_lexicon_latest.xlsx"
TOPIC_LEXICON_FILE = BASE_DIR / "topic_lexicon_seeds_only.xlsx"
OUTPUT_SENTENCE_LEVEL = BASE_DIR / "absa_sentence_level.xlsx"
OUTPUT_REVIEW_LEVEL = BASE_DIR / "absa_review_level.xlsx"

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_000_000

# echtes 1–5 Sterne Sentimentmodell
SENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
sent_tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL)
sent_model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL)


# ------------------------------------------------------------
# 1. Topic-Wörter (Fallback)
# ------------------------------------------------------------

TOPIC_WORDS = {
    "FOOD": {
        "food", "meal", "sandwich", "chicken", "fries", "nuggets", "salad",
        "wrap", "strips", "biscuit", "muffin", "hash browns", "mac and cheese",
        "soup", "sauce", "dressing", "taste", "flavor", "crispy", "fresh",
        "spicy", "portion", "quality", "temperature", "cold food", "hot food"
    },
    "DRINKS": {
        "drink", "drinks", "beverage", "soda", "tea", "sweet tea", "iced tea",
        "lemonade", "diet lemonade", "sunjoy", "water", "juice", "orange juice",
        "milk", "chocolate milk", "coffee", "iced coffee", "hot coffee",
        "milkshake", "shake", "frosted lemonade", "refill", "ice"
    },
    "HYGIENE": {
        "clean", "dirty", "hygiene", "messy", "filthy", "spotless", "sanitary",
        "unsanitary", "cleanliness", "sticky", "smell", "odor", "trash",
        "garbage", "bathroom", "restroom", "table", "floor", "counter", "sink"
    },
    "SERVICE": {
        "service", "staff", "employee", "employees", "worker", "cashier",
        "manager", "team", "friendly", "helpful", "polite", "kind", "rude",
        "attentive", "welcoming", "professional", "unprofessional", "attitude",
        "customer service", "greeted"
    },
    "VALUE": {
        "price", "prices", "expensive", "cheap", "value", "cost", "overpriced",
        "affordable", "worth", "deal", "combo", "coupon", "discount", "charged",
        "receipt", "portion size", "reasonable", "unreasonable", "money"
    },
    "AMBIENCE": {
        "ambience", "atmosphere", "vibe", "environment", "seating", "seat",
        "table", "dining room", "inside", "restaurant", "location", "crowded",
        "quiet", "loud", "noisy", "comfortable", "decor", "music", "lighting"
    },
    "ACCESSIBILITY": {
        "wheelchair", "accessible", "accessibility", "entrance", "ramp", "stairs",
        "disabled", "handicap", "elevator", "restroom access", "parking access",
        "easy access", "door", "space", "mobility"
    },
    "PARKING": {
        "parking", "parking lot", "parking spot", "parking spaces", "car", "cars",
        "lot", "garage", "parked", "hard to park", "easy parking", "traffic",
        "entrance", "exit", "street parking", "curbside"
    },
    "SPEED": {
        "fast", "slow", "quick", "quickly", "speed", "speedy", "delay",
        "delayed", "instant", "efficient", "inefficient", "wait", "waiting",
        "wait time", "long wait", "short wait", "line", "queue", "long line",
        "short line", "rush", "busy", "took forever", "took too long"
    }
}


# ------------------------------------------------------------
# 2. Rating-Lexikon laden
# ------------------------------------------------------------

def load_rating_lexicon(path):
    df = pd.read_excel(path)
    lex = {}
    for _, row in df.iterrows():
        term = str(row["text"]).lower().strip()
        rating = int(row["rating"])
        conf = float(row["confidence"])
        lex[term] = (rating, conf)
    return lex


# ------------------------------------------------------------
# 3. Topic-Lexikon laden
# ------------------------------------------------------------

def load_topic_lexicon(path):
    df = pd.read_excel(path)
    lex = {}
    for _, row in df.iterrows():
        term = str(row["text"]).lower().strip()
        topic = str(row["name_cluster"])
        conf = float(row["confidence"])
        lex[term] = (topic, conf)
    return lex


# ------------------------------------------------------------
# 4. echtes BERT-Sentiment (1–5 Sterne)
# ------------------------------------------------------------

def bert_sentence_rating(sent):
    text = sent.text
    inputs = sent_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = sent_model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]
    rating = int(np.argmax(probs) + 1)
    confidence = float(probs.max())
    return rating, confidence


# ------------------------------------------------------------
# 5. Rating-Kombination: Lexikon + BERT
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 6. Topic bestimmen
# ------------------------------------------------------------

def get_topic(term, topic_lex):
    term_norm = term.lower().strip()

    # 1. Topic-Lexikon
    if term_norm in topic_lex:
        topic, conf = topic_lex[term_norm]
        return topic, conf

    # 2. TOPIC_WORDS
    for topic, words in TOPIC_WORDS.items():
        if term_norm in words:
            return topic, 1.0

    # 3. Fallback
    return "GENERAL", 0.0


# ------------------------------------------------------------
# 7. Aspekte extrahieren (NOUN + compound)
# ------------------------------------------------------------

def extract_aspects(sent):
    aspects = []
    for token in sent:
        if token.pos_ == "NOUN":
            base = token.lemma_.lower()
            compounds = [child.text.lower() for child in token.children if child.dep_ == "compound"]
            if compounds:
                full = " ".join(compounds + [token.text.lower()])
                aspects.append(full)
            else:
                aspects.append(base)
    return list(set(aspects))


# ------------------------------------------------------------
# 8. ABSA-Satzanalyse
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 9. Review-Pipeline (mit Aggregation)
# ------------------------------------------------------------

def run_absa_pipeline(df_reviews, rating_lex, topic_lex):
    sentence_rows = []
    review_rows = []

    for _, row in df_reviews.iterrows():
        text = row["text"]
        user_id = row["user_id"]
        store_id = row["store_id"]
        time_raw = row["time"]

        try:
            dt = datetime.strptime(str(time_raw), "%d.%m.%Y %H:%M:%S")
            time_str = dt.isoformat()
        except:
            time_str = str(time_raw)

        doc = nlp(text)

        # --- Satzanalyse ---
        all_aspects = []

        for sent in doc.sents:
            sent_doc = sent.as_doc()
            sent_results = analyze_sentence(sent_doc, rating_lex, topic_lex)

            for r in sent_results:
                r["user_id"] = user_id
                r["store_id"] = store_id
                r["time"] = time_str
                sentence_rows.append(r)
                all_aspects.append(r)

        # --- Aggregation auf Review-Level ---
        topic_groups = defaultdict(list)

        for asp in all_aspects:
            topic_groups[asp["topic"]].append((asp["rating"], asp["confidence"]))

        review_result = {
            "user_id": user_id,
            "store_id": store_id,
            "time": time_str
        }

        for topic in list(TOPIC_WORDS.keys()) + ["GENERAL"]:
            vals = topic_groups.get(topic, [])
            if not vals:
                review_result[topic] = None
            else:
                ratings = np.array([v[0] for v in vals])
                confs = np.array([v[1] for v in vals])
                score = np.sum(ratings * confs) / np.sum(confs)
                review_result[topic] = score

        review_rows.append(review_result)

    return pd.DataFrame(sentence_rows), pd.DataFrame(review_rows)


# ------------------------------------------------------------
# 10. MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    df_reviews = pd.read_excel(REVIEWS_FILE)
    rating_lex = load_rating_lexicon(RATING_LEXICON_FILE)
    topic_lex = load_topic_lexicon(TOPIC_LEXICON_FILE)

    df_sentences, df_reviews_agg = run_absa_pipeline(df_reviews, rating_lex, topic_lex)

    df_sentences.to_excel(OUTPUT_SENTENCE_LEVEL, index=False)
    df_reviews_agg.to_excel(OUTPUT_REVIEW_LEVEL, index=False)

    print("ABSA Export abgeschlossen.")
    print(f"Satz-Level-Datei: {OUTPUT_SENTENCE_LEVEL}")
    print(f"Review-Level-Datei: {OUTPUT_REVIEW_LEVEL}")

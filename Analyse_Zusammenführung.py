# ============================================================
# ABSA-Sentiment-Pipeline (mit echtem BERT, Rating-Lexikon,
# Topic-Lexikon, Satzanalyse und Review-Aggregation)
# ============================================================

from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re

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

REVIEWS_FILE = DATA_DIR / "yelp_final.csv"
RATING_LEXICON_FILE = BASE_DIR / "generated_sentiment_lexicon_latest.xlsx"
TOPIC_LEXICON_FILE = BASE_DIR / "topic_lexicon_seeds_only.xlsx"
OUTPUT_SENTENCE_LEVEL = BASE_DIR / "absa_sentence_level.xlsx"
OUTPUT_REVIEW_LEVEL = BASE_DIR / "absa_review_level.xlsx"

TEXT_COLUMN = "text"
BUSINESS_NAME_COLUMN = "business_name"

CFA_PATTERNS = [
    r"chick[\s\-]?fil[\s\-]?a",
    r"chickfila",
    r"chik[\s\-]?fil[\s\-]?a",
    r"cfa\b"
]
CFA_REGEX = re.compile("|".join(CFA_PATTERNS), flags=re.IGNORECASE)

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_000_000

# echtes 1–5 Sterne Sentimentmodell
SENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
sent_tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL)
sent_model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL)


# ------------------------------------------------------------
# 1. Topic-Wörter (Fallback aus den Topic-Seeds)
# ------------------------------------------------------------

TOPIC_WORDS = {
    "FOOD": {
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
    },
    "SERVICE": {
        "service", "staff", "employee", "employees", "worker", "workers",
        "cashier", "manager", "friendly", "helpful", "polite", "rude",
        "attentive", "unprofessional", "customer service", "greeted",
        "welcoming", "attitude", "respectful", "disrespectful",
        "kind", "courteous", "patient", "impatient", "professional",
        "pleasant", "smiling", "smile", "care", "caring", "hospitality",
        "team member", "crew", "server", "assistance", "help", "ignored",
        "apologized", "accommodating", "knowledgeable", "training"
    },
    "SPEED": {
        "fast", "slow", "quick", "quickly", "speed", "speedy",
        "service speed", "order ready", "ready fast", "took forever",
        "took too long", "delay", "delayed", "immediate", "efficient",
        "inefficient", "wait", "waiting", "wait time", "long wait",
        "short wait", "line", "queue", "long line", "short line",
        "stood in line", "standing in line", "crowded line", "busy", "rush",
        "lunch rush", "dinner rush", "peak hour", "served quickly",
        "served fast", "ready quickly", "slow service", "fast service",
        "order time", "pickup time", "turnaround", "backed up"
    },
    "HYGIENE": {
        "clean", "dirty", "messy", "filthy", "spotless", "sanitary",
        "unsanitary", "hygiene", "cleanliness", "sticky", "smell",
        "odor", "trash", "garbage", "bathroom", "restroom",
        "table", "tables", "floor", "floors", "counter", "counters",
        "sink", "toilet", "napkin", "spill", "spilled", "greasy",
        "dusty", "stain", "stained", "cleaned", "unclean", "neat",
        "tidy", "sanitized", "soap", "paper towel", "overflowing trash"
    },
    "VALUE": {
        "price", "prices", "expensive", "cheap", "value", "worth",
        "deal", "combo", "meal deal", "affordable", "overpriced",
        "cost", "charged", "fair price", "reasonable", "unreasonable",
        "portion size", "money", "receipt", "bill", "total", "discount",
        "coupon", "reward", "rewards", "points", "free item", "promotion",
        "special", "pricey", "low price", "high price", "good value",
        "bad value", "not worth", "worth it", "small portion", "large portion"
    },
    "AMBIENCE": {
        "ambience", "atmosphere", "vibe", "environment", "seating",
        "dining room", "inside", "restaurant", "location", "crowded",
        "quiet", "loud", "noisy", "comfortable", "uncomfortable",
        "decor", "music", "lighting", "space", "layout", "interior",
        "booth", "chair", "chairs", "table", "tables", "family friendly",
        "kid friendly", "play area", "temperature", "air conditioning",
        "warm", "cold", "busy atmosphere", "calm", "relaxing", "modern"
    },
    "PARKING": {
        "parking", "parking lot", "parking spot", "parking spaces",
        "car", "cars", "lot", "garage", "parked", "hard to park",
        "easy parking", "traffic", "entrance", "exit", "street parking",
        "curbside", "curb", "pickup spot", "parking area", "small lot",
        "full lot", "crowded lot", "traffic flow", "turn in", "turn out",
        "blocked", "congested", "nearby parking", "free parking"
    },
    "DRINKS": {
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
    },
    "ACCESSIBILITY": {
        "wheelchair", "accessible", "accessibility", "entrance",
        "ramp", "stairs", "disabled", "handicap", "elevator",
        "restroom access", "parking access", "easy access", "door",
        "automatic door", "wide door", "narrow door", "mobility", "walker",
        "stroller", "step", "curb", "sidewalk", "path", "space",
        "accessible parking", "handicap parking", "accessible table",
        "accessible restroom", "low counter", "high counter"
    },
    "GENERAL": {
        "good", "bad", "great", "nice", "amazing", "terrible",
        "awful", "excellent", "fine", "okay", "average",
        "experience", "visit", "place", "spot", "location",
        "recommend", "disappointed", "satisfied", "overall", "favorite",
        "best", "worst", "love", "liked", "enjoyed", "happy", "unhappy",
        "return", "come back", "never again", "consistent", "inconsistent"
    }
}


# ------------------------------------------------------------
# 2. Input und Lexika laden
# ------------------------------------------------------------

def load_cfa_reviews(path):
    df = pd.read_csv(path)

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

    if df_cfa.empty:
        raise ValueError("Keine Chick-fil-A-Bewertungen in yelp_final.csv gefunden.")

    return df_cfa


def row_value(row, columns, default=None):
    for column in columns:
        if column in row and pd.notna(row[column]):
            return row[column]
    return default


def load_rating_lexicon(path):
    df = pd.read_excel(path)
    lex = {}
    for _, row in df.iterrows():
        term = str(row["text"]).lower().strip()
        rating = int(row["rating"])
        conf = float(row["confidence"])
        lex[term] = (rating, conf)
    return lex


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
# 3. echtes BERT-Sentiment (1–5 Sterne)
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
# 4. Rating-Kombination: Lexikon + BERT
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
# 5. Topic bestimmen
# ------------------------------------------------------------

def get_topic(term, topic_lex):
    term_norm = term.lower().strip()

    # 1. Topic-Lexikon
    if term_norm in topic_lex:
        topic, conf = topic_lex[term_norm]
        return topic, conf

    # 2. Fallback mit Topic-Seeds
    for topic, words in TOPIC_WORDS.items():
        if term_norm in words:
            return topic, 1.0

    # 3. Fallback
    return "GENERAL", 0.0


# ------------------------------------------------------------
# 6. Aspekte extrahieren (NOUN + compound)
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
# 7. ABSA-Satzanalyse
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
# 8. Review-Pipeline (mit Aggregation)
# ------------------------------------------------------------

def run_absa_pipeline(df_reviews, rating_lex, topic_lex):
    sentence_rows = []
    review_rows = []

    for _, row in df_reviews.iterrows():
        text = row[TEXT_COLUMN]
        user_id = row_value(row, ["user_id", "author_id", "reviewer_id"])
        store_id = row_value(row, ["store_id", "business_id", "location_id", BUSINESS_NAME_COLUMN])
        time_raw = row_value(row, ["time", "date", "created_at", "review_date"])

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

        for topic in list(TOPIC_WORDS.keys()):
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
# 9. MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    df_reviews = load_cfa_reviews(REVIEWS_FILE)
    rating_lex = load_rating_lexicon(RATING_LEXICON_FILE)
    topic_lex = load_topic_lexicon(TOPIC_LEXICON_FILE)

    df_sentences, df_reviews_agg = run_absa_pipeline(df_reviews, rating_lex, topic_lex)

    df_sentences.to_excel(OUTPUT_SENTENCE_LEVEL, index=False)
    df_reviews_agg.to_excel(OUTPUT_REVIEW_LEVEL, index=False)

    print("ABSA Export abgeschlossen.")
    print(f"Verwendete Chick-fil-A-Bewertungen: {len(df_reviews)}")
    print(f"Satz-Level-Datei: {OUTPUT_SENTENCE_LEVEL}")
    print(f"Review-Level-Datei: {OUTPUT_REVIEW_LEVEL}")

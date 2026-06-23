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

TOPIC_WORDS = {topic: {str(seed).lower().strip() for seed in seeds} for topic, seeds in TOPIC_SEEDS.items()}

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_000_000

SENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
sent_tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL)
sent_model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL)


def load_cfa_reviews(path):
    print("Lade Yelp-Daten...")
    df = pd.read_csv(path)
    print(f"Geladene Zeilen: {len(df):,}")
    missing_columns = {TEXT_COLUMN, BUSINESS_NAME_COLUMN} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Diese Spalten fehlen in yelp_final.csv: {sorted(missing_columns)}")
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
        lex[str(row["text"]).lower().strip()] = (int(row["rating"]), float(row["confidence"]))
    print(f"Rating-Lexikon-Terme: {len(lex):,}")
    return lex


def load_topic_lexicon(path):
    print("Lade Topic-Lexikon...")
    df = pd.read_excel(path)
    lex = {}
    for _, row in df.iterrows():
        lex[str(row["text"]).lower().strip()] = (str(row["name_cluster"]), float(row["confidence"]))
    print(f"Topic-Lexikon-Terme: {len(lex):,}")
    return lex


def bert_sentence_rating(sent):
    inputs = sent_tokenizer(sent.text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = sent_model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]
    return int(np.argmax(probs) + 1), float(probs.max())


def combine_rating(term, sent, rating_lex):
    term_norm = term.lower().strip()
    bert_rating, bert_conf = bert_sentence_rating(sent)
    if term_norm in rating_lex:
        lex_rating, lex_conf = rating_lex[term_norm]
        alpha = max(0.0, min(1.0, lex_conf))
        return alpha * lex_rating + (1 - alpha) * bert_rating, alpha * lex_conf + (1 - alpha) * bert_conf, "lexicon+bert"
    return bert_rating, bert_conf, "bert_only"


def get_topic(term, topic_lex):
    term_norm = term.lower().strip()
    if term_norm in topic_lex:
        return topic_lex[term_norm]
    for topic, words in TOPIC_WORDS.items():
        if term_norm in words:
            return topic, 1.0
    return "GENERAL", 0.0


def extract_aspects(sent):
    aspects = []
    for token in sent:
        if token.pos_ == "NOUN":
            compounds = [child.text.lower() for child in token.children if child.dep_ == "compound"]
            aspects.append(" ".join(compounds + [token.text.lower()]) if compounds else token.lemma_.lower())
    return list(set(aspects))


def analyze_sentence(sent, rating_lex, topic_lex):
    aspects = extract_aspects(sent)
    results = []
    if not aspects:
        rating, conf, src = combine_rating(sent.text, sent, rating_lex)
        return [{"sentence": sent.text, "text": sent.text, "rating": rating, "confidence": conf, "topic": "GENERAL", "topic_confidence": 0.0, "source": src}]
    for asp in aspects:
        rating, conf, src = combine_rating(asp, sent, rating_lex)
        topic, topic_conf = get_topic(asp, topic_lex)
        results.append({"sentence": sent.text, "text": asp, "rating": rating, "confidence": conf, "topic": topic, "topic_confidence": topic_conf, "source": src})
    return results


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
            time_str = datetime.strptime(str(time_raw), "%d.%m.%Y %H:%M:%S").isoformat()
        except Exception:
            time_str = str(time_raw)
        doc = nlp(text)
        all_aspects = []
        for sent in doc.sents:
            for r in analyze_sentence(sent.as_doc(), rating_lex, topic_lex):
                r["user_id"] = user_id
                r["store_id"] = store_id
                r["time"] = time_str
                sentence_rows.append(r)
                all_aspects.append(r)
        topic_groups = defaultdict(list)
        for asp in all_aspects:
            topic_groups[asp["topic"]].append((asp["rating"], asp["confidence"]))
        review_result = {"user_id": user_id, "store_id": store_id, "time": time_str}
        for topic in list(TOPIC_WORDS.keys()):
            vals = topic_groups.get(topic, [])
            review_result[topic] = None if not vals else np.sum(np.array([v[0] for v in vals]) * np.array([v[1] for v in vals])) / np.sum(np.array([v[1] for v in vals]))
        review_rows.append(review_result)
    print(f"Satz-Level-Zeilen: {len(sentence_rows):,}")
    print(f"Review-Level-Zeilen: {len(review_rows):,}")
    return pd.DataFrame(sentence_rows), pd.DataFrame(review_rows)


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

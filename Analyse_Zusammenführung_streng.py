# ============================================================
# Strengere ABSA-Variante
# - Originales ABSA-Skript bleibt unveraendert
# - Topic-Zuordnung aus dem Topic-Lexikon braucht hoehere Confidence
# - Analysiert zusaetzlich ChickFilA_Bereinigt 2.xlsx als Washington-Datei
# ============================================================

import importlib

import pandas as pd

absa = importlib.import_module("Analyse_Zusammenführung")

STRICT_TOPIC_MIN_CONFIDENCE = 0.80

YELP_REVIEWS_FILE = absa.DATA_DIR / "yelp_final.csv"
WASHINGTON_REVIEWS_FILE = absa.DATA_DIR / "ChickFilA_Bereinigt 2.xlsx"
STRICT_TOPIC_LEXICON_FILE = absa.DATA_DIR / "Themenlexikon.xlsx"

YELP_OUTPUT_SENTENCE_LEVEL = absa.BASE_DIR / "absa_sentence_level_streng.xlsx"
YELP_OUTPUT_REVIEW_LEVEL = absa.BASE_DIR / "absa_review_level_streng.xlsx"

WASHINGTON_OUTPUT_SENTENCE_LEVEL = absa.BASE_DIR / "absa_sentence_level_washington.xlsx"
WASHINGTON_OUTPUT_REVIEW_LEVEL = absa.BASE_DIR / "absa_review_level_washington.xlsx"


# ============================================================
# Strengere Topic-Zuordnung
# ============================================================

def get_topic_strict(term, topic_lex):
    term_norm = term.lower().strip()

    if term_norm in topic_lex:
        topic, conf = topic_lex[term_norm]
        if conf >= STRICT_TOPIC_MIN_CONFIDENCE:
            return topic, conf

    for topic, words in absa.TOPIC_WORDS.items():
        if term_norm in words:
            return topic, 1.0

    return "GENERAL", 0.0


# ============================================================
# Input laden
# ============================================================

def read_reviews(path):
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_reviews(path, dataset_name):
    print(f"Lade {dataset_name}: {path}")
    df = read_reviews(path)
    print(f"Geladene Zeilen ({dataset_name}): {len(df):,}")

    if absa.TEXT_COLUMN not in df.columns:
        raise ValueError(f"Diese Spalte fehlt in {path.name}: {absa.TEXT_COLUMN}")

    df = df[df[absa.TEXT_COLUMN].notna()].copy()
    df[absa.TEXT_COLUMN] = df[absa.TEXT_COLUMN].astype(str)
    df = df[df[absa.TEXT_COLUMN].str.strip() != ""]
    df = df[df[absa.TEXT_COLUMN].str.strip().str.lower() != "nan"]

    if absa.BUSINESS_NAME_COLUMN in df.columns:
        is_cfa = df[absa.BUSINESS_NAME_COLUMN].astype(str).str.contains(absa.CFA_REGEX, na=False)
        df = df[is_cfa].copy()
        print(f"Gefundene Chick-fil-A-Bewertungen ({dataset_name}): {len(df):,}")
    else:
        print(f"Keine business_name-Spalte in {path.name}; Datei wird als bereits bereinigt behandelt.")

    if df.empty:
        raise ValueError(f"Keine auswertbaren Bewertungen in {path.name} gefunden.")

    return df


# ============================================================
# Lauf + Export
# ============================================================

def run_and_export(dataset_name, reviews_file, output_sentence_level, output_review_level, rating_lex, topic_lex):
    df_reviews = load_reviews(reviews_file, dataset_name)
    df_sentences, df_reviews_agg = absa.run_absa_pipeline(df_reviews, rating_lex, topic_lex)

    print(f"Speichere Outputs fuer {dataset_name}...")
    df_sentences.to_excel(output_sentence_level, index=False)
    df_reviews_agg.to_excel(output_review_level, index=False)

    print(f"Fertig ({dataset_name}).")
    print(f"Verwendete Bewertungen: {len(df_reviews):,}")
    print(f"Satz-Level-Datei: {output_sentence_level}")
    print(f"Review-Level-Datei: {output_review_level}")


if __name__ == "__main__":
    absa.get_topic = get_topic_strict

    rating_lex = absa.load_rating_lexicon(absa.RATING_LEXICON_FILE)
    topic_lex = absa.load_topic_lexicon(STRICT_TOPIC_LEXICON_FILE)

    run_and_export(
        "Yelp streng",
        YELP_REVIEWS_FILE,
        YELP_OUTPUT_SENTENCE_LEVEL,
        YELP_OUTPUT_REVIEW_LEVEL,
        rating_lex,
        topic_lex,
    )

    run_and_export(
        "Washington streng",
        WASHINGTON_REVIEWS_FILE,
        WASHINGTON_OUTPUT_SENTENCE_LEVEL,
        WASHINGTON_OUTPUT_REVIEW_LEVEL,
        rating_lex,
        topic_lex,
    )

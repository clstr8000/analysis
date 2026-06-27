from pathlib import Path
from datetime import datetime
from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# =========================================================
# KONFIGURATION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LEXICON_FILE = DATA_DIR / "yelp_final.csv"

TEXT_COLUMN = "text"
RATING_COLUMN = "rating"

MAX_ROWS = 50_000
SOURCE_ROW_LIMIT = 2_000_000
CSV_CHUNKSIZE = 200_000
RANDOM_STATE = 42
LEXICON_MIN_DF = 10
LEXICON_MAX_FEATURES = 20_000
LEXICON_OUTPUT_TERMS = 20_000
NGRAM_RANGE = (1, 2)

TIMESTAMP = datetime.now().strftime("%d_%m_%Y_%H_%M")

OUTPUT_LEXICON_MODEL = BASE_DIR / "sentiment_lexicon_model.joblib"
OUTPUT_LEXICON_EXCEL = BASE_DIR / f"generated_sentiment_lexicon_{TIMESTAMP}.xlsx"
OUTPUT_LEXICON_LATEST = BASE_DIR / "generated_sentiment_lexicon_latest.xlsx"


# =========================================================
# HILFSFUNKTIONEN
# =========================================================

def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def sample_csv(path, encoding):
    rng = np.random.default_rng(RANDOM_STATE)
    sample_df = pd.DataFrame()
    source_rows_seen = 0

    reader = pd.read_csv(
        path,
        encoding=encoding,
        usecols=[TEXT_COLUMN, RATING_COLUMN],
        chunksize=CSV_CHUNKSIZE
    )

    for chunk in reader:
        source_rows_seen += len(chunk)
        chunk = chunk[[TEXT_COLUMN, RATING_COLUMN]].copy()
        chunk[TEXT_COLUMN] = chunk[TEXT_COLUMN].map(normalize_text)
        chunk[RATING_COLUMN] = pd.to_numeric(chunk[RATING_COLUMN], errors="coerce")
        chunk = chunk[(chunk[TEXT_COLUMN] != "") & chunk[RATING_COLUMN].between(1, 5)]

        if not chunk.empty:
            chunk["_sample_key"] = rng.random(len(chunk))
            sample_df = pd.concat([sample_df, chunk], ignore_index=True)
            if len(sample_df) > MAX_ROWS:
                sample_df = sample_df.nsmallest(MAX_ROWS, "_sample_key").reset_index(drop=True)

        if SOURCE_ROW_LIMIT is not None and source_rows_seen >= SOURCE_ROW_LIMIT:
            break

    if sample_df.empty:
        return sample_df.drop(columns=["_sample_key"], errors="ignore")

    sample_df = sample_df.drop(columns=["_sample_key"])
    return sample_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def read_lexicon_basis(path):
    for encoding in ("utf-8-sig", "latin1"):
        try:
            return sample_csv(path, encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, "Datei konnte nicht gelesen werden.")


# =========================================================
# LEXIKON ERSTELLEN
# =========================================================

def build_rating_lexicon(lexicon_source_df):
    print("Bereite Texte und Ratings vor...")

    lexicon_df = lexicon_source_df[[TEXT_COLUMN, RATING_COLUMN]].copy()

    if lexicon_df.empty:
        raise ValueError("Keine gültigen Texte mit Ratings gefunden.")

    print(f"Verwendete Trainingszeilen: {len(lexicon_df):,}")
    print("Erzeuge 1- bis 2-Gramm-Matrix...")

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=None,
        ngram_range=NGRAM_RANGE,
        min_df=LEXICON_MIN_DF,
        max_features=LEXICON_MAX_FEATURES,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]{2,}\b",
    )

    X = vectorizer.fit_transform(lexicon_df[TEXT_COLUMN])
    y = lexicon_df[RATING_COLUMN].astype(int)
    print(f"Matrix erstellt: {X.shape[0]:,} Zeilen x {X.shape[1]:,} Features")

    print("Trainiere schnelle logistische Regression...")
    logreg = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=300,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    logreg.fit(X, y)

    print("Wähle wichtigste Features aus...")
    feature_scores = np.max(np.abs(logreg.coef_), axis=0)
    n_terms = min(LEXICON_OUTPUT_TERMS, X.shape[1])
    selected_indices = np.argsort(feature_scores)[-n_terms:]
    selected_indices = selected_indices[np.argsort(feature_scores[selected_indices])[::-1]]

    terms = vectorizer.get_feature_names_out()[selected_indices]
    print(f"Ausgewählte Lexikon-Terme: {len(terms):,}")

    print("Berechne Rating und Confidence pro Term...")
    X_terms = csr_matrix(
        (
            np.ones(len(selected_indices)),
            (np.arange(len(selected_indices)), selected_indices)
        ),
        shape=(len(selected_indices), X.shape[1])
    )

    probs = logreg.predict_proba(X_terms)
    ratings = logreg.classes_[probs.argmax(axis=1)]
    confidences = probs.max(axis=1)

    lexicon = pd.DataFrame({
        "text": terms,
        "rating": ratings,
        "confidence": confidences
    })

    lexicon = lexicon.sort_values("confidence", ascending=False)

    reduced_vectorizer = CountVectorizer(
        vocabulary=terms,
        lowercase=True,
        ngram_range=NGRAM_RANGE,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]{2,}\b",
    )

    return reduced_vectorizer, lexicon


# =========================================================
# MAIN
# =========================================================

def main():
    if not LEXICON_FILE.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {LEXICON_FILE.resolve()}")

    print("Lade zufällige Lexikonbasis...")

    lexicon_basis_df = read_lexicon_basis(LEXICON_FILE)
    print(f"Gezogene Trainingszeilen: {len(lexicon_basis_df):,}")

    missing_lexicon_columns = {TEXT_COLUMN, RATING_COLUMN} - set(lexicon_basis_df.columns)

    if missing_lexicon_columns:
        raise ValueError(
            f"Diese Spalten fehlen in yelp_final.csv: "
            f"{sorted(missing_lexicon_columns)}"
        )

    print("Erstelle domänenspezifisches Rating-Lexikon...")

    vectorizer, lexicon = build_rating_lexicon(lexicon_basis_df)

    print("Speichere Lexikon-Artefakt...")

    artifact = {
        "vectorizer": vectorizer,
        "lexicon": lexicon,
        "text_column": TEXT_COLUMN,
        "rating_column": RATING_COLUMN,
        "max_rows": MAX_ROWS,
        "source_row_limit": SOURCE_ROW_LIMIT,
        "random_state": RANDOM_STATE,
        "lexicon_min_df": LEXICON_MIN_DF,
        "lexicon_max_features": LEXICON_MAX_FEATURES,
        "lexicon_output_terms": LEXICON_OUTPUT_TERMS,
        "ngram_range": NGRAM_RANGE,
        "solver": "lbfgs",
        "created_at": TIMESTAMP,
    }

    joblib.dump(artifact, OUTPUT_LEXICON_MODEL)

    print("Speichere Lexikon-Excel-Dateien...")

    with pd.ExcelWriter(OUTPUT_LEXICON_EXCEL, engine="openpyxl") as writer:
        lexicon.to_excel(writer, sheet_name="lexicon", index=False)

    with pd.ExcelWriter(OUTPUT_LEXICON_LATEST, engine="openpyxl") as writer:
        lexicon.to_excel(writer, sheet_name="lexicon", index=False)

    print("\nFERTIG\n")
    print(f"Lexikon-Artefakt: {OUTPUT_LEXICON_MODEL}")
    print(f"Lexikon-Excel-Datei: {OUTPUT_LEXICON_EXCEL}")
    print(f"Aktuelle Lexikon-Excel-Datei: {OUTPUT_LEXICON_LATEST}")


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    main()

from pathlib import Path
from datetime import datetime
from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel


# =========================================================
# KONFIGURATION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LEXICON_FILE = DATA_DIR / "yelp_final.csv"

TEXT_COLUMN = "text"
RATING_COLUMN = "rating"

MAX_ROWS = 200_000
RANDOM_STATE = 42
LEXICON_MIN_DF = 5
LEXICON_MAX_FEATURES = 50_000   # maximale Anzahl der Features

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


def read_lexicon_basis(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


# =========================================================
# LEXIKON ERSTELLEN
# =========================================================

def build_rating_lexicon(lexicon_source_df):
    print("Bereite Texte und Ratings vor...")

    lexicon_df = lexicon_source_df[[TEXT_COLUMN, RATING_COLUMN]].copy()

    lexicon_df[TEXT_COLUMN] = lexicon_df[TEXT_COLUMN].map(normalize_text)

    lexicon_df[RATING_COLUMN] = pd.to_numeric(
        lexicon_df[RATING_COLUMN],
        errors="coerce"
    )

    lexicon_df = lexicon_df[
        (lexicon_df[TEXT_COLUMN] != "")
        & lexicon_df[RATING_COLUMN].between(1, 5)
    ].copy()

    if lexicon_df.empty:
        raise ValueError("Keine gültigen Texte mit Ratings gefunden.")

    if len(lexicon_df) > MAX_ROWS:
        print(f"Ziehe zufällige Stichprobe mit {MAX_ROWS:,} von {len(lexicon_df):,} Zeilen.")
        lexicon_df = lexicon_df.sample(n=MAX_ROWS, random_state=RANDOM_STATE).copy()

    print(f"Verwendete Trainingszeilen: {len(lexicon_df):,}")
    print("Erzeuge 1- bis 3-Gramm-Matrix...")

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=None,
        ngram_range=(1, 3),   # 1–3-Wort-Ausdrücke
        min_df=LEXICON_MIN_DF,
        max_features=LEXICON_MAX_FEATURES,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]{2,}\b",
    )

    X = vectorizer.fit_transform(lexicon_df[TEXT_COLUMN])
    y = lexicon_df[RATING_COLUMN].astype(int)
    print(f"Matrix erstellt: {X.shape[0]:,} Zeilen x {X.shape[1]:,} Features")

    print("Trainiere logistische Regression...")
    logreg = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=2000,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    logreg.fit(X, y)

    print("Selektiere wichtige Features...")
    selector = SelectFromModel(
        logreg,
        max_features=LEXICON_MAX_FEATURES,
        prefit=True
    )

    mask = selector.get_support()
    selected_indices = mask.nonzero()[0]

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
        ngram_range=(1, 3),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]{2,}\b",
    )

    return reduced_vectorizer, lexicon


# =========================================================
# MAIN
# =========================================================

def main():
    if not LEXICON_FILE.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {LEXICON_FILE.resolve()}")

    print("Lade Lexikonbasis...")

    lexicon_basis_df = read_lexicon_basis(LEXICON_FILE)
    print(f"Geladene Zeilen: {len(lexicon_basis_df):,}")

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
        "random_state": RANDOM_STATE,
        "lexicon_min_df": LEXICON_MIN_DF,
        "lexicon_max_features": LEXICON_MAX_FEATURES,
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

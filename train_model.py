"""Train the SpamSense model: preprocessing -> TF-IDF -> Multinomial Naive Bayes."""
import json, sys
import joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from preprocess import preprocess

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/spam.csv"

def load(path):
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    cols = {c.lower(): c for c in df.columns}
    if "v1" in cols and "v2" in cols:              # Kaggle SMS Spam Collection
        df = df[[cols["v1"], cols["v2"]]]
    elif "label" in cols and "text" in cols:
        df = df[[cols["label"], cols["text"]]]
    elif "category" in cols and "message" in cols:
        df = df[[cols["category"], cols["message"]]]
    else:
        df = df.iloc[:, :2]
    df.columns = ["label", "text"]
    df = df.dropna().drop_duplicates()
    df["label"] = df["label"].astype(str).str.lower().str.strip().map(
        lambda x: 1 if x in ("spam", "1") else 0)
    return df

df = load(DATA)
print(f"Loaded {len(df)} emails  (spam={df.label.sum()}, ham={(df.label==0).sum()})")
df["clean"] = df["text"].apply(preprocess)

X_tr, X_te, y_tr, y_te = train_test_split(df["clean"], df["label"], test_size=0.2,
                                          random_state=42, stratify=df["label"])
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)
Xtr = vec.fit_transform(X_tr); Xte = vec.transform(X_te)
clf = MultinomialNB(alpha=0.1).fit(Xtr, y_tr)

pred = clf.predict(Xte)
metrics = {
    "accuracy": round(accuracy_score(y_te, pred), 4),
    "precision": round(precision_score(y_te, pred), 4),
    "recall": round(recall_score(y_te, pred), 4),
    "f1": round(f1_score(y_te, pred), 4),
    "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
    "train_size": int(len(X_tr)), "test_size": int(len(X_te)),
}
print(classification_report(y_te, pred, target_names=["Ham", "Spam"]))
print("Confusion matrix [[TN FP],[FN TP]]:", metrics["confusion_matrix"])

joblib.dump({"vectorizer": vec, "model": clf}, "model/spamsense.joblib")
json.dump(metrics, open("model/metrics.json", "w"), indent=2)
print("Saved model/spamsense.joblib and model/metrics.json")

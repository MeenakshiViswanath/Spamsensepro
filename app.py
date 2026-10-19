"""SpamSense - NLP-based intelligent email filtering system (Flask backend)."""
import json, os
import joblib
from flask import Flask, render_template, request, jsonify
from preprocess import preprocess
import db

app = Flask(__name__)
BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "model", "spamsense.joblib")

if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found. Run:  python make_dataset.py && python train_model.py")

bundle = joblib.load(MODEL_PATH)
vectorizer, model = bundle["vectorizer"], bundle["model"]
db.init_db()

def classify(text):
    X = vectorizer.transform([preprocess(text)])
    proba = model.predict_proba(X)[0]
    spam_p = float(proba[1])
    label = "Spam" if spam_p >= 0.5 else "Ham"
    conf = spam_p if label == "Spam" else 1 - spam_p
    # top words pushing towards spam, for explainability
    names = vectorizer.get_feature_names_out()
    idx = X.nonzero()[1]
    log_ratio = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    contrib = sorted(((names[i], float(log_ratio[i] * X[0, i])) for i in idx), key=lambda t: -abs(t[1]))[:6]
    keywords = [{"word": w, "signal": "spam" if s > 0 else "ham"} for w, s in contrib]
    return label, round(conf * 100, 2), round(spam_p * 100, 2), keywords

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="Please enter email text."), 400
    label, conf, spam_p, keywords = classify(text)
    db.save_prediction(text, label, conf)
    return jsonify(label=label, confidence=conf, spam_probability=spam_p, keywords=keywords)

@app.route("/api/history")
def api_history():
    return jsonify(history=db.get_history(), stats=db.get_stats())

@app.route("/api/history/clear", methods=["POST"])
def api_clear():
    db.clear_history()
    return jsonify(ok=True)

@app.route("/api/metrics")
def metrics():
    p = os.path.join(BASE, "model", "metrics.json")
    return jsonify(json.load(open(p)) if os.path.exists(p) else {})

if __name__ == "__main__":
    app.run(debug=True)

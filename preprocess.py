"""NLP preprocessing for SpamSense: clean -> tokenize -> stop-word removal -> stemming."""
import re

try:
    import nltk
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
    try:
        from nltk.corpus import stopwords
        STOP_WORDS = set(stopwords.words("english"))
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            STOP_WORDS = set(stopwords.words("english"))
        except Exception:
            raise ImportError
except ImportError:  # fallback so the app still runs without NLTK data
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOP_WORDS = set(ENGLISH_STOP_WORDS)
    class _NoStem:
        def stem(self, w): return w
    _stemmer = _NoStem()

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NUM_RE = re.compile(r"\d+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str) -> str:
    tokens = clean_text(text).split()
    tokens = [_stemmer.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

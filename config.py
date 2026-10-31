import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "default-secret")
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

    # Kredensial Telegram (opsional — Telethon user session)
    # api_id & api_hash gratis dari https://my.telegram.org
    TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID", "")
    TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "")

    # Instagram (opsional — Instaloader; disarankan akun kedua)
    INSTAGRAM_USERNAME = os.getenv("INSTAGRAM_USERNAME", "")
    INSTAGRAM_PASSWORD = os.getenv("INSTAGRAM_PASSWORD", "")
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

    # Path dataset & model
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    STATIC_DIR = os.path.join(BASE_DIR, "static")
    CHARTS_DIR = os.path.join(STATIC_DIR, "charts")
    TELEGRAM_SESSION = os.path.join(DATA_DIR, "telegram_session.session")

    COMMENTS_CSV = os.path.join(DATA_DIR, "komentar_media.csv")
    VIDEOS_CSV = os.path.join(DATA_DIR, "video_sumber.csv")

    NB_MODEL = os.path.join(MODEL_DIR, "naive_bayes.pkl")
    SVM_MODEL = os.path.join(MODEL_DIR, "svm.pkl")
    TFIDF_MODEL = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    METRICS_JSON = os.path.join(MODEL_DIR, "metrics.json")

    # Parameter crawling
    SEARCH_QUERIES = ["kebijakan pemerintah", "layanan publik"]
    MAX_VIDEOS_PER_QUERY = 5
    MAX_COMMENT_PAGES_PER_VIDEO = 3


os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.CHARTS_DIR, exist_ok=True)

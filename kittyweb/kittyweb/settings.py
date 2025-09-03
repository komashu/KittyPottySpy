from pathlib import Path
import os

#

BASE_DIR = Path(__file__).resolve().parent.parent

def get_env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)

DEBUG = get_env("DJANGO_DEBUG", "1") in {"1", "true", "True", "yes"}

SECRET_KEY = get_env("DJANGO_SECRET_KEY")
if not SECRET_KEY:
    if DEBUG:
        # Safe placeholder for dev/build time; real key must be provided in prod
        SECRET_KEY = "dev-placeholder-not-secret"
    else:
        raise RuntimeError("Missing required environment variable: DJANGO_SECRET_KEY")

# ---- Development safety warning ----
if DEBUG and SECRET_KEY == "dev-placeholder-not-secret":
    import sys
    print(
        "⚠️  WARNING: Django is running in DEBUG mode with a placeholder SECRET_KEY.",
        file=sys.stderr,
    )
    print(
        "   Set DJANGO_SECRET_KEY in your .env and disable DJANGO_DEBUG before deploying to production.",
        file=sys.stderr,
    )

ALLOWED_HOSTS = [host.strip() for host in get_env("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",") if host.strip()]
CSRF_TRUSTED_ORIGINS = [origin.strip() for origin in get_env("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",") if origin.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "reviewer",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "kittyweb.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "kittyweb.wsgi.application"

META_DIR = BASE_DIR / "meta"
META_DIR.mkdir(exist_ok=True)
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(META_DIR / "feedback.sqlite"),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = get_env("DJANGO_TIME_ZONE", "UTC")
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "static_collected"
STATICFILES_DIRS = [BASE_DIR / "static"] if (BASE_DIR / "static").exists() else []

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

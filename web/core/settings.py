"""
Django settings for DPM web project.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-dpm-dev-key-change-in-production",
)

DEBUG = os.environ.get("DJANGO_DEBUG", "True").lower() in ("true", "1", "yes")

ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")

# HF Spaces 透過 iframe 嵌入，需設定 CSRF 信任
CSRF_TRUSTED_ORIGINS = os.environ.get(
    "CSRF_TRUSTED_ORIGINS",
    "https://*.hf.space,https://*.huggingface.co",
).split(",")

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.staticfiles",
    "prediction",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",   # <-- 靜態檔案（production）
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "prediction.middleware.PasswordProtectMiddleware",
]

ROOT_URLCONF = "core.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "core.wsgi.application"

# 使用輕量級 SQLite（僅用於 session）
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# Session（儲存預測結果供下載用）
SESSION_ENGINE = "django.contrib.sessions.backends.db"

LANGUAGE_CODE = "zh-hant"
TIME_ZONE = "Asia/Taipei"
USE_I18N = True
USE_TZ = True

# ── 靜態檔案 ──
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

# ── HF Spaces iframe 嵌入 ──
X_FRAME_OPTIONS = "ALLOWALL"

# ── 密碼保護 ──
DPM_ACCESS_PASSWORD = os.environ.get("DPM_ACCESS_PASSWORD", "dpm2026")
# 完整授權密碼：使用此密碼登入可上傳無限筆批次資料
DPM_FULL_PASSWORD = os.environ.get("DPM_FULL_PASSWORD", "")

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Utilise une image Python légère
FROM python:3.12-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de dépendances puis installer
COPY requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libfreetype6-dev \
        libpng-dev \
        libjpeg-dev \
        libopenjp2-7-dev \
        libtiff5-dev \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
        libharfbuzz-dev \
        libfribidi-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip wheel setuptools==68.0.0
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . /app

# Exposer le port utilisé par Flask
EXPOSE 5000

# Variables d'environnement pour Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Commande de démarrage
CMD ["python", "app.py"]

FROM tensorflow/tensorflow:2.10.0-gpu

WORKDIR /app

# install dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# install directly via pip
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    pandas \
    matplotlib \
    scipy \
    scikit-learn \
    networkx \
    seaborn \
    sqlalchemy \
    flask \
    alembic \
    pyyaml \
    tqdm \
    pillow \
    cloudpickle \
    bcrypt \
    spacy==3.7.5 \
    spacy-lookups-data \
    mlflow \
    cryptography

# copy code into image
COPY . .

RUN python3 -m spacy download en_core_web_sm

CMD ["python3", "_hyperparameterTuning.py"]

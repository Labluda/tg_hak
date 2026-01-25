from fastapi import FastAPI
import pandas as pd
from catboost import CatBoostRegressor
import gdown
from utils import data_transform, features_for_test

# ------------------------------
# Настройка FastAPI
# ------------------------------
app = FastAPI(title="CatBoost Prediction API")

# ------------------------------
# Загрузка модели с Google Drive
# ------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1XAtCA6S9KMqJ0h2Ud4BvF-OFqZKtxiwZ"
MODEL_PATH = "best_catboost_model.cbm"

gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = CatBoostRegressor()
model.load_model(MODEL_PATH)

# ------------------------------
# Эндпоинт для предсказаний
# ------------------------------
@app.post("/predict")
def predict(data: dict):
    """
    data: dict с входными данными
    Пример: {"cpm": [1.2, 2.3], "weekday": [1, 5], "channel_name": ["A", "B"], ...}
    """
    df = pd.DataFrame(data)
    df = data_transform(df)
    df = features_for_test(df)

    # Убираем колонки, которые не нужны для модели
    if 'views' in df.columns:
        df = df.drop(['views', 'date'], axis=1, errors='ignore')

    preds = model.predict(df)
    return {"predictions": preds.tolist()}
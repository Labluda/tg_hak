from fastapi import FastAPI, Request
import pandas as pd
from catboost import CatBoostRegressor
import gdown
from utils import data_transform, features_for_test

app = FastAPI(title="CatBoost Prediction API")

# ------------------------------
# Загрузка модели с Google Drive
# ------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1XAtCA6S9KMqJ0h2Ud4BvF-OFqZKtxiwZ"
MODEL_PATH = "trained_model/best_catboost_model.cbm"

gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = CatBoostRegressor()
model.load_model(MODEL_PATH)

# ------------------------------
# Эндпоинт для предсказаний
# ------------------------------
@app.post("/predict")
async def predict(request: Request):
    """
    Принимает JSON вида:
    {"cpm": float, "channel": str, "data": str}
    Возвращает {"predicted_views": int}
    """
    json_data = await request.json()

    # Преобразуем JSON в DataFrame с одной строкой
    df = pd.DataFrame([json_data])

    # Переименуем колонку 'channel' в 'channel_name' и 'data' в 'date' для совместимости
    if 'channel' in df.columns:
        df = df.rename(columns={'channel': 'channel_name'})
    if 'data' in df.columns:
        df = df.rename(columns={'data': 'date'})

    # Обработка данных
    df = data_transform(df)
    df = features_for_test(df)

    # Убираем лишние колонки перед подачей в модель
    df = df.drop(['views', 'date'], axis=1, errors='ignore')

    # Прогноз
    pred = model.predict(df)

    return {"predicted_views": int(pred[0])}

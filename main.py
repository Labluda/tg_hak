from fastapi import FastAPI, Request, UploadFile, File
import pandas as pd
from catboost import CatBoostRegressor
import gdown
from utils import data_transform, features_for_test
import os
import numpy as np
import json

# ------------------------------
# Создание папки для модели
# ------------------------------
os.makedirs("trained_model", exist_ok=True)

# ------------------------------
# Ссылка на модель и путь для сохранения
# ------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1XAtCA6S9KMqJ0h2Ud4BvF-OFqZKtxiwZ"
MODEL_PATH = "trained_model/best_catboost_model.cbm"

# ------------------------------
# Скачивание модели
# ------------------------------
gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ------------------------------
# Загрузка модели CatBoost
# ------------------------------
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

# ------------------------------
# Создание FastAPI приложения
# ------------------------------
app = FastAPI(title="CatBoost Prediction API")

# ------------------------------
# Эндпоинт для предсказаний JSON или JSON-файла
# ------------------------------
@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile | None = File(default=None)
):
    """
    Эндпоинт принимает:
    1) raw JSON:
       [
         {"cpm": 1.2, "channel": "A", "data": "2026-01-25"}
       ]
    2) JSON-файл через браузер
    Возвращает массив предсказаний:
       [{"predicted_views": 123}, ...]
    """
    # ------------------------------
    # Чтение данных
    # ------------------------------
    if file is not None:
        # Если пришёл файл, читаем как JSON
        contents = await file.read()
        try:
            json_data = json.loads(contents)
        except json.JSONDecodeError:
            return {"error": "Файл не является корректным JSON"}
    else:
        # Иначе читаем тело запроса как JSON
        try:
            json_data = await request.json()
        except json.JSONDecodeError:
            return {"error": "Тело запроса пустое или невалидный JSON"}

    # ------------------------------
    # Преобразуем в DataFrame
    # ------------------------------
    df = pd.DataFrame(json_data)

    # Переименовываем колонки для совместимости
    if 'channel' in df.columns:
        df = df.rename(columns={'channel': 'channel_name'})
    if 'data' in df.columns:
        df = df.rename(columns={'data': 'date'})

    # ------------------------------
    # Обработка признаков
    # ------------------------------
    df = data_transform(df)
    df = features_for_test(df)
    df = df.drop(['views', 'date'], axis=1, errors='ignore')

    # ------------------------------
    # Прогноз
    # ------------------------------
    log_preds = model.predict(df)
    preds = np.expm1(log_preds)

    return [{"predicted_views": int(p)} for p in preds]

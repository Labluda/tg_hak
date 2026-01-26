from fastapi import FastAPI, Request
import pandas as pd
from catboost import CatBoostRegressor
import gdown
from utils import data_transform, features_for_test
import os

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
# Эндпоинт для предсказаний массива JSON
# ------------------------------
@app.post("/predict")
async def predict(request: Request):
    """
    Эндпоинт принимает массив JSON объектов, например:
    [
        {"cpm": 1.2, "channel": "A", "data": "2026-01-25"},
        {"cpm": 2.3, "channel": "B", "data": "2026-01-25"}
    ]
    Возвращает массив предсказаний:
    [
        {"predicted_views": 123},
        {"predicted_views": 456}
    ]
    """
    # Получаем JSON
    json_data = await request.json()

    # Преобразуем в DataFrame
    df = pd.DataFrame(json_data)

    # Переименуем колонки для совместимости с функциями подготовки
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
    log_preds = model.predict(df)
    preds = np.expm1(log_preds)


    # Формируем массив JSON с предсказаниями
    return [{"predicted_views": int(p)} for p in preds]

# CatBoost FastAPI Prediction Service

Веб-сервис на **FastAPI** для предсказания просмотров (`views`) рекламных объявлений на основе модели **CatBoost**.  

Модель хранится на **Google Drive** и скачивается автоматически при старте сервиса.  

Сервис работает онлайн на **Render** и принимает **массив JSON объектов** для батч-предсказаний.

---

## Структура проекта

project_folder/

│

├─ main.py # FastAPI веб-сервис, обрабатывает запросы и делает предсказания

├─ requirements.txt # зависимости Python

├─ utils.py # функции подготовки данных и создания признаков

├─ notebooks/ # ноутбуки с обучением и анализом

│ └─ h_reg_optuna.ipynb

├─ trained_model/ # создаётся автоматически, сюда скачивается модель

├─ data/ # исходные данные

│ ├─ train.csv # обучающий датасет

│ ├─ old_test.csv # старый тестовый датасет

│ └─ new_test.csv # новый тестовый датасет

└─ output/ # результаты предсказаний

├─ old_output.csv

└─ new_output.csv


---

## Запуск проекта на Render

1. Создайте **Web Service** на Render с Python.  
2. Укажите команды:  

- **Build Command:**
```bash
pip install -r requirements.txt

Start Command:
uvicorn main:app --host 0.0.0.0 --port $PORT
Render автоматически создаёт виртуальное окружение, скачивает модель с Google Drive и запускает сервис.

Входные данные

Сервис принимает массив JSON объектов, например:
[
    {"cpm": 1.2, "channel": "A", "data": "2026-01-12"},
    {"cpm": 1.3, "channel": "A", "data": "2026-01-13"},
    {"cpm": 1.1, "channel": "A", "data": "2026-01-14"},
    ...
]

Ограничения и важные замечания

Для корректного расчёта признаков trend и seasonal через seasonal_decompose необходимо минимум 14 наблюдений на канал (2 полных недельных цикла).

Если прислать меньше данных, прогноз может быть некорректным или использоваться только историческая информация из модели.

Массив JSON объектов должен содержать хотя бы 14 записей для одного канала для корректной работы признаков сезонности.

Формат ответа

Сервис возвращает массив JSON объектов с прогнозами:
[
    {"predicted_views": 123},
    {"predicted_views": 234},
    {"predicted_views": 345}
]

Пример запроса на Python
import requests

url = "https://tg-hak-catboost.onrender.com/predict"

test_data = [
    {"cpm": 1.2, "channel": "A", "data": "2026-01-12"},
    {"cpm": 1.3, "channel": "A", "data": "2026-01-13"},
    {"cpm": 1.1, "channel": "A", "data": "2026-01-14"},
    # минимум 14 объектов на канал для корректной работы seasonal_decompose
]

response = requests.post(url, json=test_data)
print(response.json())

Подготовка признаков

Сервис автоматически создаёт признаки:

year, month, weekday, day из даты.

is_weekend — бинарный признак выходного дня.

trend и seasonal — тренд и сезонная компонента CPM через seasonal_decompose.

Дополнительно используются лаги и скользящее среднее по каналам.

Зависимости (requirements.txt)

fastapi
uvicorn
pandas
numpy
catboost
gdown
scikit-learn
optuna
seaborn
matplotlib
statsmodels
phik

Тест и отладка

Используйте Swagger UI для проверки запросов:
https://tg-hak-catboost.onrender.com/docs

Папка trained_model создаётся автоматически при старте сервиса, модель скачивается с Google Drive.

Сервис рассчитан на массив JSON объектов. Для корректного расчёта сезонности нужно минимум 14 объектов.

---

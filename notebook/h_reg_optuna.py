# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%pip install phik -q
%pip install catboost -q
import phik
from statsmodels.tsa.seasonal import seasonal_decompose
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, root_mean_squared_error
import numpy as np
import copy
%pip install optuna -q
import optuna

# %%
data = pd.read_csv('../data/train.csv')
new_test_data = pd.read_csv('../data/new_test.csv')
old_test_data = pd.read_csv('../data/old_test.csv')

# %% [markdown]
# ## Обзор

# %%
display(data.head(5))
display(data.info())
display(data.describe())

data.columns = data.columns.str.lower()
data.columns = data.columns.str.strip()

for i in data.columns:
    print(i, len(data[i].unique()))

# %% [markdown]
# ## Анализ

# %%
for i in ['cpm', 'views', 'clicks', 'actions']:
    plt.figure(figsize=(7, 3))
    sns.histplot(data=data, x=i, log_scale=(True, False))
    plt.show()

# %%
for i in ['cpm', 'views', 'clicks', 'actions']:
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=data, y=i)
    plt.show()

# %% [markdown]
# Характерным для всех данных является наличие значительного числа экстремальных значений

# %%
#Процент отрезаемых данных
len(data.query('cpm >= 100 or views >= 10**5 or clicks >= 100 or actions >= 100'))/len(data)

# %%
#Количество отрезаемых данных
len(data.query('cpm >= 100 or views >= 10**4 or clicks >= 100 or actions >= 100'))

# %%
data = data.query('cpm <= 100 and views <= 10**4 and clicks <= 100 and actions <= 100')

# %%
for i in ['cpm', 'views', 'clicks', 'actions']:
    plt.figure(figsize=(7, 3))
    sns.histplot(data=data, x=i, log_scale=(True, False))
    plt.show()

# %%
for i in ['cpm', 'views', 'clicks', 'actions']:
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=data, y=i)
    plt.show()

# %%
data['cat_num'] = pd.factorize(data['channel_name'])[0]
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['weekday'] = data['date'].dt.weekday
data['day'] = data['date'].dt.day
data

# %%
int_col = ['cpm', 'views', 'clicks', 'actions']
cat_col = ['channel_name', 'cat_num', 'year', 'month', 'weekday', 'day']
phik_matrix = data.drop(['channel_name', 'date', 'ad_id'], axis=1).phik_matrix(interval_cols = int_col)
display(phik_matrix)

# %%
plt.figure(figsize=(7, 6))
sns.heatmap(phik_matrix, annot=True, cmap='Blues')
plt.show()

# %%
data = data.drop('cat_num', axis=1)
data = data.sort_values(by='date')
test = data.copy()
test = test.set_index('date')
test = test[['views', 'cpm', 'clicks', 'actions']]

# %%
# 1. Отбор нужного периода
df_period = test['2025-06-01':'2025-07-01']

# 2. Суммирование по дням (если есть несколько записей на день)
daily_views = df_period['views'].resample('D').sum()

# 3. Декомпозиция на тренд, сезонность, остаток
decomp = seasonal_decompose(daily_views, model='additive', period=7)  # недельный сезон

# 4. График
plt.figure(figsize=(10,6))
plt.plot(daily_views.index, daily_views.values, label='Daily Views', marker='o')
plt.plot(decomp.seasonal.index, decomp.seasonal.values, label='Seasonal Component', linestyle='--')
plt.title('Views per Day (June 1-14, 2025)')
plt.xlabel('Date')
plt.ylabel('Views')
plt.legend()
plt.show()


# %%
daily_views = test['views'].resample('D').sum()

# 2. Декомпозиция (аддитивная модель)
# period=7 для недельного цикла; можно подбирать для других сезонностей
decomp = seasonal_decompose(daily_views, model='additive', period=7)

# 3. График
plt.figure(figsize=(14,6))
sns.lineplot(x=daily_views.index, y=daily_views.values, label='Daily Views')
sns.lineplot(x=decomp.trend.index, y=decomp.trend.values, label='Trend', linestyle='--')
sns.lineplot(x=decomp.seasonal.index, y=decomp.seasonal.values, label='Seasonal Component', linestyle=':')
plt.title('Daily Views with Trend and Seasonal Component')
plt.xlabel('Date')
plt.ylabel('Views')
plt.legend()
plt.show()

# %% [markdown]
# ## Создание признаков

# %%
time_data = data.copy()

# обрезка данных
time_data = time_data.query('cpm <= 100 and views <= 10**4 and clicks <= 100 and actions <= 100')
# 1. сезонные компоненты
daily_cpm = (
    time_data
    .groupby('date', as_index=False)
    .agg(cpm_mean=('cpm', 'mean'))
    .sort_values('date')
    .set_index('date')
    .asfreq('D')
)

# интерполяция пропусков
daily_cpm['cpm_mean'] = daily_cpm['cpm_mean'].interpolate(method='time')

decomp = seasonal_decompose(
    daily_cpm['cpm_mean'],
    model='additive',
    period=7,
    extrapolate_trend='freq'
)

daily_cpm['trend'] = decomp.trend
daily_cpm['seasonal'] = decomp.seasonal

daily_cpm = daily_cpm.reset_index()

time_data = time_data.merge(
    daily_cpm[['date', 'trend', 'seasonal']],
    on='date',
    how='left'
)
# 2. Лаг + скользящее среднее по каналам
daily_channel_cpm = (
    time_data
    .groupby(['date', 'channel_name'], as_index=False)
    .agg(cpm_mean=('cpm', 'mean'))
    .sort_values(['channel_name', 'date'])
)

daily_channel_cpm['cpm_lag_1'] = (
    daily_channel_cpm
    .groupby('channel_name')['cpm_mean']
    .shift(1)
)

daily_channel_cpm['cpm_rm_3'] = (
    daily_channel_cpm
    .groupby('channel_name')['cpm_mean']
    .shift(1)
    .rolling(3, min_periods=1)
    .mean()
)

# Мёрдж обратно
time_data = time_data.merge(
    daily_channel_cpm[['date', 'channel_name', 'cpm_lag_1', 'cpm_rm_3']],
    on=['date', 'channel_name'],
    how='left'
)

# 3. Убираем строки с NaN (первые дни каналов)
time_data = time_data.dropna()

time_data['is_weekend'] = np.where(time_data['weekday'] >= 5, 1, 0)
time_data['cpm_resid'] = time_data['cpm'] - time_data['trend'] - time_data['seasonal']

display(time_data)

# %% [markdown]
# ## Обучение

# %%
def objective(trial):

    params = {
        'loss_function': 'RMSE',
        'iterations': 2000,
        'verbose': False,
        'early_stopping_rounds': 50,

        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
        'depth': trial.suggest_int('depth', 3, 8),

        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 50.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),

        'random_seed': 42
    }

    cv_results = cv(
        params=params,
        pool=data_pool,
        fold_count=3,
        shuffle=True,
        partition_random_seed=42,
        verbose=False
    )

    best_rmse = cv_results['test-RMSE-mean'].min()

    return best_rmse

# %%
X = time_data.drop(['ad_id', 'clicks', 'actions', 'views', 'date'], axis=1)
y = np.log1p(time_data['views'])
data_pool = Pool(data=X, label=y, cat_features = ['channel_name', 'is_weekend'])

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(
    objective,
    n_trials=20,
    show_progress_bar=True,
    n_jobs=-1
)

print("Best RMSE:", study.best_value)
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# %%
best_params = study.best_trial.params

best_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=best_params['learning_rate'],
    depth=best_params['depth'],
    l2_leaf_reg=best_params['l2_leaf_reg'],
    min_data_in_leaf=best_params['min_data_in_leaf'],
    loss_function='RMSE',
    verbose=False
)
best_model.fit(X, y, cat_features=['channel_name', 'is_weekend'])

# %%
best_model.save_model('../trained_model/best_catboost_model.cbm')
loaded_model = CatBoostRegressor()
loaded_model.load_model('../trained_model/best_catboost_model.cbm')

# %% [markdown]
# ## Тест

# %%
def data_transform(df):
    df.columns = df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday
    df['day'] = df['date'].dt.day

# %%
def features_for_test(df):
    daily_cpm = (
        df
        .groupby('date', as_index=False)
        .agg(cpm_mean=('cpm', 'mean'))
        .sort_values('date')
        .set_index('date')
        .asfreq('D')
    )

    daily_cpm['cpm_mean'] = daily_cpm['cpm_mean'].interpolate(method='time')

    decomp = seasonal_decompose(
        daily_cpm['cpm_mean'],
        model='additive',
        period=7,
        extrapolate_trend='freq'
    )

    daily_cpm['trend'] = decomp.trend
    daily_cpm['seasonal'] = decomp.seasonal
    daily_cpm = daily_cpm.reset_index()

    df = df.merge(
        daily_cpm[['date', 'trend', 'seasonal']],
        on='date',
        how='left'
    )

    daily_channel_cpm = (
        df
        .groupby(['date', 'channel_name'], as_index=False)
        .agg(cpm_mean=('cpm', 'mean'))
        .sort_values(['channel_name', 'date'])
    )

    daily_channel_cpm['cpm_lag_1'] = (
        daily_channel_cpm
        .groupby('channel_name')['cpm_mean']
        .shift(1)
    )

    daily_channel_cpm['cpm_rm_3'] = (
        daily_channel_cpm
        .groupby('channel_name')['cpm_mean']
        .shift(1)
        .rolling(3, min_periods=1)
        .mean()
    )

    df = df.merge(
        daily_channel_cpm[['date', 'channel_name', 'cpm_lag_1', 'cpm_rm_3']],
        on=['date', 'channel_name'],
        how='left'
    )

    df['is_weekend'] = (
        (df['weekday'] >= 5)
        .fillna(False)
        .astype('int32')
    )

    df['cpm_resid'] = df['cpm'] - df['trend'] - df['seasonal']

    return df

# %%
data_transform(old_test_data)
data_transform(new_test_data)
old_test_data = features_for_test(old_test_data)
new_test_data = features_for_test(new_test_data)

# %%
y_pred_new=loaded_model.predict(new_test_data.drop(['date', 'views'], axis=1))
y_pred_old=loaded_model.predict(old_test_data.drop(['date', 'views'], axis=1))

# %%
y_pred_new = np.expm1(y_pred_new)
y_pred_old = np.expm1(y_pred_old)

# %%
final_new_data = pd.read_csv('../data/new_test.csv')
final_old_data = pd.read_csv('../data/old_test.csv')
final_new_data['VIEWS'] = y_pred_new
final_old_data['VIEWS'] = y_pred_old
final_new_data.to_csv('../output/new_output.csv')
final_old_data.to_csv('../output/old_output.csv')



import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def data_transform(df: pd.DataFrame) -> pd.DataFrame:
    # Приведение всех колонок к нижнему регистру
    df.columns = df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday
    df['day'] = df['date'].dt.day
    return df

def features_for_test(df: pd.DataFrame) -> pd.DataFrame:
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

    df['is_weekend'] = ((df['weekday'] >= 5).astype(int))
    df['cpm_resid'] = df['cpm'] - df['trend'] - df['seasonal']

    return df
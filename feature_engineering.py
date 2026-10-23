                       
from __future__ import annotations

import os
from typing import Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

DEVICE_FILES = {
    'weather': 'Train_Test_IoT_Weather.csv',
    'modbus': 'Train_Test_IoT_Modbus.csv',
    'thermostat': 'Train_Test_IoT_Thermostat.csv',
    'fridge': 'Train_Test_IoT_Fridge.csv',
    'garage_door': 'Train_Test_IoT_Garage_Door.csv',
    'gps': 'Train_Test_IoT_GPS_Tracker.csv',
    'motion_light': 'Train_Test_IoT_Motion_Light.csv',
}
DEVICE_ORDER = (
    'weather', 'modbus', 'thermostat', 'fridge',
    'garage_door', 'gps', 'motion_light',
)
WEATHER_FEATS = (
    'weather_temperature',
    'weather_pressure',
    'weather_humidity',
)
META_DROP = ('date', 'time', 'timestamp', 'label', 'type')
OUTPUT_CSV = 'iot_fusion_weather_all.csv'


def load_device_table(path: str, device_name: str) -> pd.DataFrame:
    pass
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if 'time' in df.columns:
        df['time'] = df['time'].astype(str).str.strip()
    if 'date' in df.columns:
        df['date'] = df['date'].astype(str).str.strip()
    df['timestamp'] = pd.to_datetime(
        df['date'] + ' ' + df['time'], format='%d-%b-%y %H:%M:%S')
    feature_cols = [c for c in df.columns if c not in META_DROP]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.rename(columns={c: f'{device_name}_{c}' for c in feature_cols})
    return df


def load_all_devices(
    data_dir: str = '.',
    files: Optional[Mapping[str, str]] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    files = dict(files or DEVICE_FILES)
    dfs = {}
    for name in DEVICE_ORDER:
        filename = files[name]
        df = load_device_table(os.path.join(data_dir, filename), name)
        dfs[name] = df
        if verbose:
            print(f'[{name}] 样本数: {len(df)}, '
                  f'时间范围: {df["timestamp"].min()} ~ {df["timestamp"].max()}')
    return dfs


def _mean_num(s: pd.Series) -> float:
    return s.mean(numeric_only=True)


def _mode_or_normal(s: pd.Series) -> str:
    m = s.mode()
    return m.iloc[0] if len(m) > 0 else 'normal'


def aggregate_by_timestamp(dfs: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    pass
    weather_agg = dfs['weather'].groupby('timestamp').agg({
        **{c: _mean_num for c in WEATHER_FEATS},
        'label': 'max',
        'type': _mode_or_normal,
    }).reset_index()
    agg_dfs = {'weather': weather_agg}

    for name in DEVICE_ORDER[1:]:
        feature_cols = [c for c in dfs[name].columns if c.startswith(f'{name}_')]
        agg = dfs[name].groupby('timestamp').agg({
            **{c: _mean_num for c in feature_cols},
            'label': 'max',
            'type': _mode_or_normal,
        }).reset_index()
        agg = agg.rename(columns={
            'label': f'{name}_label',
            'type': f'{name}_type',
        })
        agg_dfs[name] = agg
    return agg_dfs


def merge_on_timestamp(agg_dfs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    pass
    merged = agg_dfs['weather']
    for name in DEVICE_ORDER[1:]:
        merged = pd.merge(
            merged, agg_dfs[name], on='timestamp', how='outer',
            suffixes=('', f'_{name}'))
    return merged


def unify_attack_labels(merged: pd.DataFrame) -> pd.DataFrame:
    pass
    merged = merged.copy()
    label_cols = [c for c in merged.columns if c.endswith('_label')] + ['label']
    type_cols = [c for c in merged.columns if c.endswith('_type')] + ['type']
    merged['final_label'] = merged[label_cols].max(axis=1)

    def get_attack_type(row):
        for c in type_cols:
            val = row[c]
            if pd.notna(val) and val != 'normal':
                return val
        return 'normal'

    merged['final_type'] = merged.apply(get_attack_type, axis=1)
    return merged


def feature_columns(merged: pd.DataFrame) -> list:
    label_cols = [c for c in merged.columns if c.endswith('_label')] + ['label']
    type_cols = [c for c in merged.columns if c.endswith('_type')] + ['type']
    skip = {'timestamp', 'final_label', 'final_type', 'label', 'type'}
    skip.update(label_cols)
    skip.update(type_cols)
    return [c for c in merged.columns if c not in skip]


def fill_missing_features(merged: pd.DataFrame) -> Tuple[pd.DataFrame, Sequence[str]]:
    pass
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    cols = feature_columns(merged)
    merged[cols] = merged[cols].ffill().fillna(0)
    return merged, cols


def build_iot_fusion(
    data_dir: str = '.',
    output_csv: Optional[str] = OUTPUT_CSV,
    save: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    pass
    dfs = load_all_devices(data_dir, verbose=verbose)
    agg_dfs = aggregate_by_timestamp(dfs)
    merged = merge_on_timestamp(agg_dfs)
    merged = unify_attack_labels(merged)
    merged, cols = fill_missing_features(merged)

    if verbose:
        print('\n===== 合并结果 =====')
        print(f'总时间步: {len(merged)}')
        print(f'特征维度: {len(cols)}')
        print(f'攻击样本占比: {merged["final_label"].mean():.2%}')
        print(f'攻击类型分布:\n{merged["final_type"].value_counts()}')

    if save:
        out = output_csv or OUTPUT_CSV
        if not os.path.isabs(out):
            out = os.path.join(data_dir, out)
        merged.to_csv(out, index=False)
        if verbose:
            print(f'\n已保存: {os.path.basename(out)}')
    return merged


if __name__ == '__main__':
    build_iot_fusion(data_dir='.')

# core/data_processor.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='linear')
        self.target_column = None
        self.datetime_column = None
        self.feature_columns = []
        
    def preprocess_data(self, df: pd.DataFrame, datetime_column: str, target_column: str, 
                       handle_missing: bool = True, remove_outliers: bool = True,
                       make_stationary: bool = True, feature_engineering: bool = True) -> pd.DataFrame:
        
        self.datetime_column = datetime_column
        self.target_column = target_column
        
        df_processed = df.copy()
        
        if datetime_column in df_processed.columns:
            df_processed[datetime_column] = pd.to_datetime(df_processed[datetime_column])
            df_processed = df_processed.set_index(datetime_column)
            df_processed = df_processed.sort_index()
        
        if handle_missing:
            df_processed = self._handle_missing_values(df_processed)
        
        if remove_outliers:
            df_processed = self._remove_outliers(df_processed, target_column)
        
        if make_stationary:
            df_processed = self._make_stationary(df_processed, target_column)
        
        if feature_engineering:
            df_processed = self._feature_engineering(df_processed)
        
        self.feature_columns = [col for col in df_processed.columns if col != target_column]
        
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                if df_processed[col].isnull().sum() / len(df_processed) > 0.3:
                    df_processed = df_processed.drop(columns=[col])
                else:
                    df_processed[col] = df_processed[col].interpolate(method='linear')
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
        
        df_processed = df_processed.dropna()
        
        return df_processed
    
    def _remove_outliers(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        df_processed = df.copy()
        
        if target_column in df_processed.columns:
            Q1 = df_processed[target_column].quantile(0.25)
            Q3 = df_processed[target_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df_processed[target_column] < lower_bound) | (df_processed[target_column] > upper_bound)
            df_processed = df_processed[~outlier_mask]
        
        return df_processed
    
    def _make_stationary(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        df_processed = df.copy()
        
        if target_column in df_processed.columns:
            from statsmodels.tsa.stattools import adfuller
            
            result = adfuller(df_processed[target_column].dropna())
            p_value = result[1]
            
            if p_value > 0.05:
                df_processed[f'{target_column}_stationary'] = df_processed[target_column].diff().fillna(0)
                
                result_diff = adfuller(df_processed[f'{target_column}_stationary'].dropna())
                if result_diff[1] <= 0.05:
                    df_processed[target_column] = df_processed[f'{target_column}_stationary']
                    df_processed = df_processed.drop(columns=[f'{target_column}_stationary'])
        
        return df_processed
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        
        if self.target_column in df_processed.columns:
            target_values = df_processed[self.target_column]
            
            df_processed[f'{self.target_column}_lag1'] = target_values.shift(1)
            df_processed[f'{self.target_column}_lag7'] = target_values.shift(7)
            df_processed[f'{self.target_column}_lag30'] = target_values.shift(30)
            
            df_processed[f'{self.target_column}_rolling_mean_7'] = target_values.rolling(window=7).mean()
            df_processed[f'{self.target_column}_rolling_std_7'] = target_values.rolling(window=7).std()
            df_processed[f'{self.target_column}_rolling_mean_30'] = target_values.rolling(window=30).mean()
            df_processed[f'{self.target_column}_rolling_std_30'] = target_values.rolling(window=30).std()
            
            df_processed['day_of_week'] = df_processed.index.dayofweek
            df_processed['day_of_month'] = df_processed.index.day
            df_processed['month'] = df_processed.index.month
            df_processed['quarter'] = df_processed.index.quarter
            df_processed['year'] = df_processed.index.year
            
            df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
            df_processed['is_month_start'] = (df_processed.index.day == 1).astype(int)
            df_processed['is_month_end'] = (df_processed.index.is_month_end).astype(int)
            
            df_processed = df_processed.dropna()
        
        return df_processed
    
    def create_sequences(self, data: np.ndarray, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
        sequences = []
        targets = []
        
        for i in range(len(data) - seq_len - pred_len + 1):
            sequences.append(data[i:i + seq_len])
            targets.append(data[i + seq_len:i + seq_len + pred_len])
        
        return np.array(sequences), np.array(targets)
    
    def scale_data(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    def inverse_scale(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
    
    def get_seasonal_decomposition(self, data: pd.Series) -> Dict[str, pd.Series]:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        decomposition = seasonal_decompose(data, model='additive', period=min(30, len(data)//2))
        
        return {
            'observed': decomposition.observed,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }
    
    def detect_anomalies(self, data: pd.Series, method: str = 'zscore') -> Dict[str, Any]:
        if method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            anomalies = z_scores > 3
            
            return {
                'anomalies': anomalies,
                'anomaly_indices': data.index[anomalies],
                'anomaly_scores': z_scores,
                'method': 'zscore'
            }
        
        elif method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomalies = (data < lower_bound) | (data > upper_bound)
            
            return {
                'anomalies': anomalies,
                'anomaly_indices': data.index[anomalies],
                'anomaly_scores': (data - data.median()).abs() / IQR,
                'method': 'iqr'
            }

class AdvancedTimeSeriesProcessor(TimeSeriesProcessor):
    def __init__(self):
        super().__init__()
        self.fourier_features = []
    
    def add_fourier_features(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        df_processed = df.copy()
        
        for period in periods:
            for k in range(1, 4):
                df_processed[f'fourier_sin_{period}_{k}'] = np.sin(2 * k * np.pi * np.arange(len(df_processed)) / period)
                df_processed[f'fourier_cos_{period}_{k}'] = np.cos(2 * k * np.pi * np.arange(len(df_processed)) / period)
                self.fourier_features.extend([f'fourier_sin_{period}_{k}', f'fourier_cos_{period}_{k}'])
        
        return df_processed
    
    def add_technical_indicators(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        df_processed = df.copy()
        
        if target_column in df_processed.columns:
            prices = df_processed[target_column]
            
            df_processed['sma_7'] = prices.rolling(window=7).mean()
            df_processed['sma_30'] = prices.rolling(window=30).mean()
            df_processed['ema_7'] = prices.ewm(span=7).mean()
            df_processed['ema_30'] = prices.ewm(span=30).mean()
            
            df_processed['rsi'] = self._calculate_rsi(prices, window=14)
            df_processed['macd'] = self._calculate_macd(prices)
            df_processed['bollinger_upper'], df_processed['bollinger_lower'] = self._calculate_bollinger_bands(prices)
            
            df_processed['momentum'] = prices.diff(5)
            df_processed['volatility'] = prices.rolling(window=20).std()
        
        return df_processed
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        return macd
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band
    
    def detect_change_points(self, data: pd.Series, method: str = 'cusum') -> Dict[str, Any]:
        if method == 'cusum':
            return self._cusum_change_detection(data)
        elif method == 'binary_segmentation':
            return self._binary_segmentation_change_detection(data)
        
        return {}
    
    def _cusum_change_detection(self, data: pd.Series) -> Dict[str, Any]:
        mean = data.mean()
        std = data.std()
        
        cusum_pos = 0
        cusum_neg = 0
        change_points = []
        
        threshold = 5 * std
        
        for i, value in enumerate(data):
            cusum_pos = max(0, cusum_pos + value - mean - std)
            cusum_neg = max(0, cusum_neg + mean - std - value)
            
            if cusum_pos > threshold or cusum_neg > threshold:
                change_points.append(i)
                cusum_pos = 0
                cusum_neg = 0
        
        return {
            'change_points': change_points,
            'change_indices': data.index[change_points] if hasattr(data, 'index') else change_points,
            'method': 'cusum'
        }
    
    def _binary_segmentation_change_detection(self, data: pd.Series) -> Dict[str, Any]:
        from scipy import stats
        
        def find_change_point(segment):
            if len(segment) < 10:
                return -1
            
            t_stat_max = 0
            change_point = -1
            
            for i in range(5, len(segment) - 5):
                left = segment[:i]
                right = segment[i:]
                
                t_stat, _ = stats.ttest_ind(left, right, equal_var=False)
                t_stat = abs(t_stat)
                
                if t_stat > t_stat_max:
                    t_stat_max = t_stat
                    change_point = i
            
            return change_point if t_stat_max > 2 else -1
        
        def recursive_segmentation(segment, start_idx, change_points):
            cp = find_change_point(segment)
            
            if cp != -1:
                global_cp = start_idx + cp
                change_points.append(global_cp)
                
                recursive_segmentation(segment[:cp], start_idx, change_points)
                recursive_segmentation(segment[cp:], global_cp, change_points)
        
        change_points = []
        recursive_segmentation(data.values, 0, change_points)
        
        return {
            'change_points': sorted(change_points),
            'change_indices': data.index[change_points] if hasattr(data, 'index') else change_points,
            'method': 'binary_segmentation'
        }
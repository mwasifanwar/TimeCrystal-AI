# core/time_series_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import math

class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, seq_len=100, pred_len=30, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.input_projection = nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model * seq_len, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, pred_len)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.input_projection(x.unsqueeze(-1))
        x = self.positional_encoding(x)
        
        encoded = self.transformer_encoder(x)
        
        encoded_flat = encoded.reshape(batch_size, -1)
        output = self.decoder(encoded_flat)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class Informer(TimeSeriesTransformer):
    def __init__(self, d_model=512, nhead=8, num_layers=3, seq_len=100, pred_len=30, dropout=0.1):
        super().__init__(d_model, nhead, num_layers, seq_len, pred_len, dropout)
        
        self.prob_attention = ProbAttention(mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False)
        self.distilling = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.input_projection(x.unsqueeze(-1))
        x = self.positional_encoding(x)
        
        x = x.transpose(1, 2)
        x = self.distilling(x)
        x = x.transpose(1, 2)
        
        encoded, _ = self.prob_attention(x, x, x)
        encoded_flat = encoded.reshape(batch_size, -1)
        output = self.decoder(encoded_flat)
        
        return output

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, queries, keys, values):
        B, L, E = queries.shape
        _, S, D = values.shape
        
        scale = self.scale or 1.0 / math.sqrt(E)
        
        scores = torch.einsum("ble,bse->bls", queries, keys)
        
        if self.mask_flag:
            if queries.shape[1] != keys.shape[1]:
                mask = torch.ones(L, S).tril().to(queries.device)
                scores.masked_fill_(mask == 0, -1e9)
        
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bls,bsd->bld", A, values)
        
        if self.output_attention:
            return V, A
        return V, None

class Autoformer(TimeSeriesTransformer):
    def __init__(self, d_model=512, nhead=8, num_layers=3, seq_len=100, pred_len=30, dropout=0.1):
        super().__init__(d_model, nhead, num_layers, seq_len, pred_len, dropout)
        
        self.seasonal_decomp = SeasonalDecomposition()
        self.autocorrelation = AutoCorrelation()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        seasonal_init, trend_init = self.seasonal_decomp(x.unsqueeze(-1))
        
        x = self.input_projection(seasonal_init)
        x = self.positional_encoding(x)
        
        encoded = self.transformer_encoder(x)
        
        seasonal_part, _ = self.autocorrelation(encoded, encoded, encoded)
        trend_part = trend_init
        
        combined = seasonal_part + trend_part
        encoded_flat = combined.reshape(batch_size, -1)
        output = self.decoder(encoded_flat)
        
        return output

class SeasonalDecomposition(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    def forward(self, x):
        trend = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend

class AutoCorrelation(nn.Module):
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, queries, keys, values):
        B, L, E = queries.shape
        _, S, D = values.shape
        
        q_fft = torch.fft.rfft(queries, dim=1)
        k_fft = torch.fft.rfft(keys, dim=1)
        
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=1)
        
        if self.mask_flag:
            mask = torch.ones(L, S).tril().to(queries.device)
            corr.masked_fill_(mask == 0, 0)
        
        V = torch.einsum("bls,bsd->bld", corr, values)
        return V, corr

class MultiHorizonPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def predict(self, model, data, horizon, enable_uncertainty=True, confidence_level=0.95, n_simulations=500):
        model.eval()
        
        if isinstance(data, pd.DataFrame):
            target_col = [col for col in data.columns if col not in ['datetime', 'date', 'timestamp']][0]
            values = data[target_col].values
        else:
            values = data
        
        seq_len = model.seq_len
        pred_len = horizon
        
        if len(values) < seq_len:
            raise ValueError(f"Data length {len(values)} is less than sequence length {seq_len}")
        
        last_sequence = values[-seq_len:].astype(np.float32)
        input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            point_forecast = model(input_tensor).cpu().numpy().flatten()
        
        results = {
            'point_forecast': pd.Series(point_forecast, index=range(len(values), len(values) + pred_len)),
            'model': model,
            'horizon': horizon
        }
        
        if enable_uncertainty:
            uncertainty_intervals = self._quantify_uncertainty(
                model, input_tensor, n_simulations, confidence_level
            )
            results['uncertainty_intervals'] = uncertainty_intervals
            results['confidence_level'] = confidence_level
        
        visualizer = TimeSeriesVisualizer()
        forecast_plot = visualizer.plot_forecast(data, results)
        results['forecast_plot'] = forecast_plot
        
        if len(values) > pred_len:
            metrics = self._calculate_metrics(values, point_forecast)
            results['metrics'] = metrics
        
        return results
    
    def multi_horizon_forecast(self, model, data, horizons=[7, 30, 90]):
        results = {}
        
        for horizon in horizons:
            horizon_results = self.predict(model, data, horizon, enable_uncertainty=True)
            results[f'horizon_{horizon}'] = horizon_results
        
        return results
    
    def _quantify_uncertainty(self, model, input_tensor, n_simulations, confidence_level):
        model.train()
        
        forecasts = []
        for _ in range(n_simulations):
            with torch.no_grad():
                forecast = model(input_tensor).cpu().numpy().flatten()
                forecasts.append(forecast)
        
        forecasts = np.array(forecasts)
        
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(forecasts, lower_percentile, axis=0)
        upper_bound = np.percentile(forecasts, upper_percentile, axis=0)
        mean_forecast = np.mean(forecasts, axis=0)
        
        return {
            'lower': pd.Series(lower_bound),
            'upper': pd.Series(upper_bound),
            'mean': pd.Series(mean_forecast),
            'all_simulations': forecasts
        }
    
    def _calculate_metrics(self, actual, predicted):
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        if len(actual) < len(predicted):
            actual = actual[-len(predicted):]
        elif len(actual) > len(predicted):
            actual = actual[:len(predicted)]
        
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

class AdvancedTimeSeriesTransformer(TimeSeriesTransformer):
    def __init__(self, d_model=512, nhead=8, num_layers=6, seq_len=100, pred_len=30, dropout=0.1, 
                 use_attention=True, use_cnn=True, use_lstm=True):
        super().__init__(d_model, nhead, num_layers, seq_len, pred_len, dropout)
        
        self.use_attention = use_attention
        self.use_cnn = use_cnn
        self.use_lstm = use_lstm
        
        if use_cnn:
            self.cnn_encoder = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(seq_len)
            )
        
        if use_lstm:
            self.lstm_encoder = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=True
            )
        
        self.attention_weights = nn.Parameter(torch.ones(3))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        features = []
        
        if self.use_cnn:
            cnn_features = self.cnn_encoder(x.unsqueeze(1)).squeeze(1)
            features.append(cnn_features)
        
        transformer_features = self.input_projection(x.unsqueeze(-1))
        transformer_features = self.positional_encoding(transformer_features)
        transformer_features = self.transformer_encoder(transformer_features)
        features.append(transformer_features.reshape(batch_size, -1))
        
        if self.use_lstm:
            lstm_features, _ = self.lstm_encoder(transformer_features)
            features.append(lstm_features.reshape(batch_size, -1))
        
        weights = F.softmax(self.attention_weights, dim=0)
        combined_features = sum(w * f for w, f in zip(weights, features))
        
        output = self.decoder(combined_features)
        return output
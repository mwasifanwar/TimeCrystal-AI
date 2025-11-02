# utils/visualization.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns

class TimeSeriesVisualizer:
    def __init__(self):
        self.colors = {
            'actual': '#2E86AB',
            'forecast': '#A23B72',
            'confidence': 'rgba(162, 59, 114, 0.2)',
            'train': '#2E86AB',
            'test': '#F18F01',
            'anomaly': '#C73E1D'
        }
    
    def plot_forecast(self, historical_data: pd.DataFrame, forecast_results: Dict[str, Any]) -> go.Figure:
        target_column = [col for col in historical_data.columns if col not in ['datetime', 'date', 'timestamp']][0]
        
        historical_values = historical_data[target_column]
        point_forecast = forecast_results['point_forecast']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=historical_values.index,
            y=historical_values.values,
            mode='lines',
            name='Historical',
            line=dict(color=self.colors['actual'], width=2)
        ))
        
        forecast_index = pd.date_range(
            start=historical_values.index[-1] + pd.Timedelta(days=1),
            periods=len(point_forecast),
            freq='D'
        )
        
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=point_forecast.values,
            mode='lines',
            name='Forecast',
            line=dict(color=self.colors['forecast'], width=2, dash='dash')
        ))
        
        if 'uncertainty_intervals' in forecast_results:
            intervals = forecast_results['uncertainty_intervals']
            
            fig.add_trace(go.Scatter(
                x=forecast_index.tolist() + forecast_index.tolist()[::-1],
                y=intervals['upper'].tolist() + intervals['lower'].tolist()[::-1],
                fill='toself',
                fillcolor=self.colors['confidence'],
                line=dict(color='rgba(255,255,255,0)'),
                name=f"{forecast_results.get('confidence_level', 0.95)*100:.0f}% Confidence"
            ))
        
        fig.update_layout(
            title='Time Series Forecast',
            xaxis_title='Date',
            yaxis_title=target_column,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_multi_horizon_forecast(self, historical_data: pd.DataFrame, 
                                  multi_forecasts: Dict[str, Any]) -> go.Figure:
        
        target_column = [col for col in historical_data.columns if col not in ['datetime', 'date', 'timestamp']][0]
        historical_values = historical_data[target_column]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=historical_values.index,
            y=historical_values.values,
            mode='lines',
            name='Historical',
            line=dict(color=self.colors['actual'], width=2)
        ))
        
        colors = ['#A23B72', '#F18F01', '#C73E1D', '#6A8EAE']
        
        for i, (horizon_name, forecast_data) in enumerate(multi_forecasts.items()):
            if horizon_name.startswith('horizon_'):
                point_forecast = forecast_data['point_forecast']
                
                forecast_index = pd.date_range(
                    start=historical_values.index[-1] + pd.Timedelta(days=1),
                    periods=len(point_forecast),
                    freq='D'
                )
                
                fig.add_trace(go.Scatter(
                    x=forecast_index,
                    y=point_forecast.values,
                    mode='lines',
                    name=f'{horizon_name.replace("horizon_", "")} days',
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        fig.update_layout(
            title='Multi-Horizon Forecast',
            xaxis_title='Date',
            yaxis_title=target_column,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_uncertainty_decomposition(self, uncertainty_results: Dict[str, Any]) -> go.Figure:
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Total Uncertainty', 'Aleatoric Uncertainty', 
                          'Epistemic Uncertainty', 'Uncertainty Growth']
        )
        
        if 'total_uncertainty' in uncertainty_results:
            total_unc = uncertainty_results['total_uncertainty']
            aleatoric = uncertainty_results.get('aleatoric_uncertainty', np.zeros_like(total_unc))
            epistemic = uncertainty_results.get('epistemic_uncertainty', np.zeros_like(total_unc))
            
            horizons = np.arange(len(total_unc))
            
            fig.add_trace(go.Scatter(
                x=horizons, y=total_unc,
                mode='lines', name='Total',
                line=dict(color='#A23B72')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=horizons, y=aleatoric,
                mode='lines', name='Aleatoric',
                line=dict(color='#F18F01')
            ), row=1, col=2)
            
            fig.add_trace(go.Scatter(
                x=horizons, y=epistemic,
                mode='lines', name='Epistemic',
                line=dict(color='#2E86AB')
            ), row=2, col=1)
        
        if 'temporal_variance' in uncertainty_results:
            temporal_var = uncertainty_results['temporal_variance']
            uncertainty_growth = np.diff(temporal_var)
            
            fig.add_trace(go.Scatter(
                x=np.arange(len(uncertainty_growth)),
                y=uncertainty_growth,
                mode='lines',
                name='Uncertainty Growth',
                line=dict(color='#C73E1D')
            ), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text='Horizon', row=2, col=1)
        fig.update_xaxes(title_text='Horizon', row=2, col=2)
        fig.update_yaxes(title_text='Uncertainty', row=1, col=1)
        fig.update_yaxes(title_text='Uncertainty', row=1, col=2)
        
        return fig
    
    def comprehensive_analysis(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        
        analysis_results = {}
        
        values = data[target_column]
        
        decomposition_plot = self.plot_seasonal_decomposition(values)
        acf_plot = self.plot_acf_pacf(values)
        distribution_plot = self.plot_distribution(values)
        
        from statsmodels.tsa.stattools import adfuller
        from scipy.stats import shapiro
        
        adf_result = adfuller(values.dropna())
        normality_result = shapiro(values.dropna())
        
        seasonal_strength = self.calculate_seasonal_strength(values)
        
        analysis_results.update({
            'decomposition_plot': decomposition_plot,
            'acf_plot': acf_plot,
            'distribution_plot': distribution_plot,
            'adf_pvalue': adf_result[1],
            'normality_pvalue': normality_result[1],
            'seasonality_strength': seasonal_strength,
            'stationary': adf_result[1] <= 0.05,
            'normal': normality_result[1] > 0.05
        })
        
        return analysis_results
    
    def plot_seasonal_decomposition(self, series: pd.Series) -> go.Figure:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        decomposition = seasonal_decompose(series, model='additive', period=min(30, len(series)//2))
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual']
        )
        
        fig.add_trace(go.Scatter(
            x=series.index, y=decomposition.observed,
            mode='lines', name='Observed'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=series.index, y=decomposition.trend,
            mode='lines', name='Trend'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=series.index, y=decomposition.seasonal,
            mode='lines', name='Seasonal'
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=series.index, y=decomposition.resid,
            mode='lines', name='Residual'
        ), row=4, col=1)
        
        fig.update_layout(height=600, showlegend=False)
        
        return fig
    
    def plot_acf_pacf(self, series: pd.Series, lags: int = 40) -> go.Figure:
        from statsmodels.tsa.stattools import acf, pacf
        
        acf_values = acf(series.dropna(), nlags=lags)
        pacf_values = pacf(series.dropna(), nlags=lags)
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=['ACF', 'PACF'])
        
        fig.add_trace(go.Bar(
            x=list(range(len(acf_values))),
            y=acf_values,
            name='ACF'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=list(range(len(pacf_values))),
            y=pacf_values,
            name='PACF'
        ), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(title_text='Lag', row=1, col=1)
        fig.update_xaxes(title_text='Lag', row=1, col=2)
        fig.update_yaxes(title_text='Correlation', row=1, col=1)
        fig.update_yaxes(title_text='Correlation', row=1, col=2)
        
        return fig
    
    def plot_distribution(self, series: pd.Series) -> go.Figure:
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Distribution', 'QQ Plot'])
        
        fig.add_trace(go.Histogram(
            x=series.values,
            nbinsx=50,
            name='Distribution',
            marker_color=self.colors['actual']
        ), row=1, col=1)
        
        from scipy.stats import probplot
        qq_data = probplot(series.dropna(), dist="norm")
        
        fig.add_trace(go.Scatter(
            x=qq_data[0][0],
            y=qq_data[0][1],
            mode='markers',
            name='QQ Plot',
            marker=dict(color=self.colors['forecast'])
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=qq_data[0][0],
            y=qq_data[1][0] * qq_data[0][0] + qq_data[1][1],
            mode='lines',
            name='Theoretical',
            line=dict(color='red', dash='dash')
        ), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False)
        
        return fig
    
    def calculate_seasonal_strength(self, series: pd.Series) -> float:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        try:
            decomposition = seasonal_decompose(series, model='additive', period=min(30, len(series)//2))
            seasonal_var = np.var(decomposition.seasonal.dropna())
            residual_var = np.var(decomposition.resid.dropna())
            
            if seasonal_var + residual_var == 0:
                return 0.0
            
            return max(0, 1 - (residual_var / (seasonal_var + residual_var)))
        except:
            return 0.0

class AdvancedTimeSeriesVisualizer(TimeSeriesVisualizer):
    def __init__(self):
        super().__init__()
    
    def plot_model_comparison(self, model_results: Dict[str, Any]) -> go.Figure:
        models = list(model_results.keys())
        metrics = ['mse', 'mae', 'rmse']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [model_results[model].get(metric, 0) for model in models]
            fig.add_trace(go.Bar(
                name=metric.upper(),
                x=models,
                y=values,
                text=[f'{v:.4f}' for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Comparison',
            xaxis_title='Models',
            yaxis_title='Metric Values',
            barmode='group',
            height=500
        )
        
        return fig
    
    def plot_anomaly_detection(self, data: pd.DataFrame, anomalies: Dict[str, Any], 
                             target_column: str) -> go.Figure:
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[target_column],
            mode='lines',
            name='Time Series',
            line=dict(color=self.colors['actual'])
        ))
        
        if 'anomaly_indices' in anomalies:
            anomaly_data = data.loc[anomalies['anomaly_indices']]
            
            fig.add_trace(go.Scatter(
                x=anomaly_data.index,
                y=anomaly_data[target_column],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color=self.colors['anomaly'],
                    size=8,
                    symbol='x'
                )
            ))
        
        fig.update_layout(
            title='Anomaly Detection',
            xaxis_title='Date',
            yaxis_title=target_column,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_forecast_accuracy(self, actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Actual vs Predicted', 'Residuals', 
                          'Residual Distribution', 'Prediction Error']
        )
        
        fig.add_trace(go.Scatter(
            x=actual, y=predicted,
            mode='markers',
            name='Predictions',
            marker=dict(color=self.colors['forecast'])
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=[actual.min(), actual.max()],
            y=[actual.min(), actual.max()],
            mode='lines',
            name='Perfect Fit',
            line=dict(color='red', dash='dash')
        ), row=1, col=1)
        
        residuals = actual - predicted
        
        fig.add_trace(go.Scatter(
            x=predicted, y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color=self.colors['actual'])
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=[predicted.min(), predicted.max()],
            y=[0, 0],
            mode='lines',
            name='Zero Line',
            line=dict(color='red', dash='dash')
        ), row=1, col=2)
        
        fig.add_trace(go.Histogram(
            x=residuals,
            nbinsx=50,
            name='Residual Distribution',
            marker_color=self.colors['confidence']
        ), row=2, col=1)
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        metrics_text = f'MAE: {mae:.4f}<br>RMSE: {rmse:.4f}'
        
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='text',
            text=[metrics_text],
            textposition='middle center',
            showlegend=False
        ), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False)
        fig.update_xaxes(title_text='Actual', row=1, col=1)
        fig.update_yaxes(title_text='Predicted', row=1, col=1)
        fig.update_xaxes(title_text='Predicted', row=1, col=2)
        fig.update_yaxes(title_text='Residuals', row=1, col=2)
        fig.update_xaxes(title_text='Residual Value', row=2, col=1)
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        fig.update_xaxes(showticklabels=False, row=2, col=2)
        fig.update_yaxes(showticklabels=False, row=2, col=2)
        
        return fig
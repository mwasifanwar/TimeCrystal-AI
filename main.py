# main.py
import streamlit as st
import torch
import pandas as pd
import numpy as np
from core.time_series_models import TimeSeriesTransformer, MultiHorizonPredictor
from core.data_processor import TimeSeriesProcessor
from core.uncertainty_quantifier import UncertaintyQuantifier
from core.model_trainer import TimeSeriesTrainer
from utils.visualization import TimeSeriesVisualizer
from utils.config import load_config

st.set_page_config(
    page_title="TimeCrystal AI - Advanced Time Series Forecasting - Wasif",
    page_icon="⏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    if 'time_series_processor' not in st.session_state:
        st.session_state.time_series_processor = None
    if 'transformer_model' not in st.session_state:
        st.session_state.transformer_model = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = {}

def load_components():
    with st.spinner("⏰ Loading TimeCrystal AI Engine..."):
        if st.session_state.time_series_processor is None:
            st.session_state.time_series_processor = TimeSeriesProcessor()
        if st.session_state.transformer_model is None:
            st.session_state.transformer_model = TimeSeriesTransformer()

def main():
    st.title("⏰ TimeCrystal AI - Advanced Time Series Forecasting")
    st.markdown("Transformer-powered time series forecasting with uncertainty quantification and multi-horizon predictions")
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("⚙️ Forecasting Configuration")
        
        forecast_type = st.selectbox(
            "Forecast Type",
            ["Univariate", "Multivariate", "Multi-Seasonal"],
            help="Select the type of time series forecasting"
        )
        
        horizon_options = {
            "Short-term": [7, 30, 90],
            "Medium-term": [90, 180, 365],
            "Long-term": [365, 730, 1095]
        }
        
        selected_horizon = st.selectbox(
            "Forecast Horizon",
            list(horizon_options.keys())
        )
        
        horizon_days = st.select_slider(
            "Days to Forecast",
            options=horizon_options[selected_horizon],
            value=horizon_options[selected_horizon][1]
        )
        
        st.subheader("Model Parameters")
        model_architecture = st.selectbox(
            "Model Architecture",
            ["Transformer", "Informer", "Autoformer", "Pyraformer"]
        )
        
        enable_uncertainty = st.checkbox("Uncertainty Quantification", value=True)
        enable_multiple_horizons = st.checkbox("Multiple Horizons", value=True)
        enable_anomaly_detection = st.checkbox("Anomaly Detection", value=True)
        
        st.subheader("Advanced Options")
        confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
        n_simulations = st.slider("Monte Carlo Simulations", 100, 1000, 500, 100)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Upload", "🤖 Model Training", "🔮 Forecasting", "📈 Analysis", "🚀 Deployment"])
    
    with tab1:
        st.header("Time Series Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload Time Series Data",
            type=['csv', 'parquet', 'xlsx', 'json'],
            help="Upload your time series data with datetime index"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                
                st.session_state.current_dataset = df
                
                st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Data Preview")
                    st.dataframe(df.head(10))
                
                with col2:
                    st.subheader("Time Series Info")
                    st.write(f"**Shape:** {df.shape}")
                    st.write(f"**Date Range:** {df.iloc[0,0]} to {df.iloc[-1,0]}")
                    st.write(f"**Observations:** {len(df)}")
                    
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    st.write(f"**Numeric Columns:** {len(numeric_cols)}")
                    
                    if len(numeric_cols) > 0:
                        st.write(f"**Mean Values:**")
                        for col in numeric_cols[:3]:
                            st.write(f"  - {col}: {df[col].mean():.2f}")
                
                st.subheader("Data Preprocessing")
                datetime_column = st.selectbox("Select Datetime Column", df.columns.tolist())
                target_column = st.selectbox("Select Target Column", numeric_cols)
                
                if st.button("🔄 Preprocess Time Series"):
                    preprocess_time_series(df, datetime_column, target_column)
                    
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
    
    with tab2:
        st.header("Model Training")
        
        if st.session_state.current_dataset is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Train Transformer Model", type="primary"):
                    train_transformer_model(model_architecture, horizon_days)
            
            with col2:
                if st.button("🔄 Train Multiple Models"):
                    train_multiple_models(horizon_days)
            
            with col3:
                if st.button("📊 Compare Models"):
                    compare_trained_models()
            
            if st.session_state.trained_models:
                display_training_results()
        else:
            st.info("📝 Please upload a time series dataset first to start training.")
    
    with tab3:
        st.header("Time Series Forecasting")
        
        if st.session_state.trained_models:
            st.subheader("Generate Forecasts")
            
            forecast_horizon = st.slider("Forecast Horizon (days)", 7, 365, horizon_days)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔮 Generate Forecast", type="primary"):
                    generate_forecasts(forecast_horizon, enable_uncertainty, confidence_level, n_simulations)
            
            with col2:
                if st.button("📈 Multiple Horizon Forecast"):
                    generate_multi_horizon_forecasts()
            
            if st.session_state.forecast_results:
                display_forecast_results()
        else:
            st.info("🤖 Train models first to generate forecasts.")
    
    with tab4:
        st.header("Time Series Analysis")
        
        if st.session_state.current_dataset is not None:
            st.subheader("Statistical Analysis")
            
            if st.button("📊 Analyze Time Series"):
                analyze_time_series()
            
            if st.session_state.forecast_results:
                st.subheader("Forecast Analysis")
                display_forecast_analysis()
        else:
            st.info("📝 Upload data to perform time series analysis.")
    
    with tab5:
        st.header("Model Deployment")
        
        if st.session_state.trained_models:
            selected_model = st.selectbox("Select Model for Deployment", list(st.session_state.trained_models.keys()))
            
            deployment_framework = st.selectbox(
                "Deployment Framework",
                ["FastAPI", "Docker Container", "AWS Lambda", "Google Cloud Functions", "Streamlit App"]
            )
            
            if st.button("🚀 Generate Deployment Code"):
                generate_deployment_code(selected_model, deployment_framework)
        else:
            st.info("🤖 Train models first to generate deployment code.")

def preprocess_time_series(df, datetime_column, target_column):
    load_components()
    
    with st.spinner("🔄 Preprocessing time series data..."):
        try:
            processed_data = st.session_state.time_series_processor.preprocess_data(
                df, 
                datetime_column, 
                target_column
            )
            st.session_state.current_dataset = processed_data
            st.success("✅ Time series preprocessing completed successfully!")
            
            st.subheader("Preprocessing Summary")
            st.write(f"**Original Shape:** {df.shape}")
            st.write(f"**Processed Shape:** {processed_data.shape}")
            st.write(f"**Missing Values Handled:** Yes")
            st.write(f"**Outliers Processed:** Yes")
            st.write(f"**Stationarity Check:** Yes")
            
        except Exception as e:
            st.error(f"❌ Time series preprocessing failed: {str(e)}")

def train_transformer_model(model_architecture, horizon_days):
    load_components()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🚀 Training transformer model...")
        
        processed_data = st.session_state.current_dataset
        target_column = st.session_state.time_series_processor.target_column
        
        trainer = TimeSeriesTrainer()
        trained_model = trainer.train_model(
            data=processed_data,
            target_column=target_column,
            model_type=model_architecture,
            forecast_horizon=horizon_days,
            epochs=100
        )
        
        st.session_state.trained_models[model_architecture] = trained_model
        
        progress_bar.progress(100)
        status_text.text("✅ Transformer model training completed successfully!")
        
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ Model training failed: {str(e)}")

def train_multiple_models(horizon_days):
    load_components()
    
    with st.spinner("🤖 Training multiple time series models..."):
        try:
            processed_data = st.session_state.current_dataset
            target_column = st.session_state.time_series_processor.target_column
            
            trainer = TimeSeriesTrainer()
            models = trainer.train_multiple_models(
                data=processed_data,
                target_column=target_column,
                forecast_horizon=horizon_days
            )
            
            st.session_state.trained_models.update(models)
            st.success(f"✅ Trained {len(models)} additional models!")
            
        except Exception as e:
            st.error(f"❌ Model training failed: {str(e)}")

def compare_trained_models():
    if not st.session_state.trained_models:
        st.warning("No trained models to compare.")
        return
    
    comparison_data = []
    for model_name, model_info in st.session_state.trained_models.items():
        comparison_data.append({
            'Model': model_name,
            'MSE': model_info.get('mse', 0),
            'MAE': model_info.get('mae', 0),
            'RMSE': model_info.get('rmse', 0),
            'Training Time': model_info.get('training_time', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.subheader("Model Comparison")
    st.dataframe(comparison_df.style.highlight_min(axis=0, color='lightgreen'))
    
    best_model = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
    st.success(f"🏆 Best Model: {best_model}")

def display_training_results():
    st.header("Training Results")
    
    for model_name, model_info in st.session_state.trained_models.items():
        with st.expander(f"📊 {model_name} - RMSE: {model_info.get('rmse', 0):.4f}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Details**")
                st.write(f"**Architecture:** {model_name}")
                st.write(f"**Training Time:** {model_info.get('training_time', 0):.2f}s")
                st.write(f"**MSE:** {model_info.get('mse', 0):.4f}")
                st.write(f"**MAE:** {model_info.get('mae', 0):.4f}")
                
            with col2:
                if 'loss_curve' in model_info:
                    st.plotly_chart(model_info['loss_curve'], use_container_width=True)

def generate_forecasts(forecast_horizon, enable_uncertainty, confidence_level, n_simulations):
    with st.spinner("🔮 Generating forecasts..."):
        try:
            best_model_name = list(st.session_state.trained_models.keys())[0]
            model_info = st.session_state.trained_models[best_model_name]
            processed_data = st.session_state.current_dataset
            
            multi_horizon_predictor = MultiHorizonPredictor()
            forecasts = multi_horizon_predictor.predict(
                model=model_info['model'],
                data=processed_data,
                horizon=forecast_horizon,
                enable_uncertainty=enable_uncertainty,
                confidence_level=confidence_level,
                n_simulations=n_simulations
            )
            
            st.session_state.forecast_results = forecasts
            st.success(f"✅ Forecasts generated for {forecast_horizon} days!")
            
        except Exception as e:
            st.error(f"❌ Forecast generation failed: {str(e)}")

def generate_multi_horizon_forecasts():
    with st.spinner("📈 Generating multi-horizon forecasts..."):
        try:
            best_model_name = list(st.session_state.trained_models.keys())[0]
            model_info = st.session_state.trained_models[best_model_name]
            processed_data = st.session_state.current_dataset
            
            multi_horizon_predictor = MultiHorizonPredictor()
            multi_forecasts = multi_horizon_predictor.multi_horizon_forecast(
                model=model_info['model'],
                data=processed_data,
                horizons=[7, 30, 90, 180]
            )
            
            st.session_state.forecast_results.update(multi_forecasts)
            st.success("✅ Multi-horizon forecasts generated!")
            
        except Exception as e:
            st.error(f"❌ Multi-horizon forecast failed: {str(e)}")

def display_forecast_results():
    st.header("Forecast Results")
    
    forecasts = st.session_state.forecast_results
    
    if 'point_forecast' in forecasts:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Point Forecast")
            st.dataframe(forecasts['point_forecast'].tail(10))
        
        with col2:
            st.subheader("Forecast Statistics")
            st.write(f"**Mean Forecast:** {forecasts['point_forecast'].mean():.2f}")
            st.write(f"**Std Forecast:** {forecasts['point_forecast'].std():.2f}")
            st.write(f"**Min Forecast:** {forecasts['point_forecast'].min():.2f}")
            st.write(f"**Max Forecast:** {forecasts['point_forecast'].max():.2f}")
    
    if 'forecast_plot' in forecasts:
        st.plotly_chart(forecasts['forecast_plot'], use_container_width=True)
    
    if 'uncertainty_intervals' in forecasts:
        st.subheader("Uncertainty Intervals")
        intervals = forecasts['uncertainty_intervals']
        st.write(f"**Lower Bound (5%):** {intervals['lower'].mean():.2f}")
        st.write(f"**Upper Bound (95%):** {intervals['upper'].mean():.2f}")
        st.write(f"**Interval Width:** {intervals['upper'].mean() - intervals['lower'].mean():.2f}")

def analyze_time_series():
    with st.spinner("📊 Analyzing time series..."):
        try:
            processed_data = st.session_state.current_dataset
            target_column = st.session_state.time_series_processor.target_column
            
            visualizer = TimeSeriesVisualizer()
            analysis_results = visualizer.comprehensive_analysis(
                data=processed_data,
                target_column=target_column
            )
            
            st.subheader("Time Series Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(analysis_results['decomposition_plot'], use_container_width=True)
            
            with col2:
                st.plotly_chart(analysis_results['acf_plot'], use_container_width=True)
            
            st.subheader("Statistical Tests")
            st.write(f"**Stationarity (ADF):** p-value = {analysis_results['adf_pvalue']:.4f}")
            st.write(f"**Normality (Shapiro-Wilk):** p-value = {analysis_results['normality_pvalue']:.4f}")
            st.write(f"**Seasonality (Strength):** {analysis_results['seasonality_strength']:.4f}")
            
        except Exception as e:
            st.error(f"❌ Time series analysis failed: {str(e)}")

def display_forecast_analysis():
    forecasts = st.session_state.forecast_results
    
    if 'metrics' in forecasts:
        st.subheader("Forecast Accuracy Metrics")
        metrics = forecasts['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MSE", f"{metrics.get('mse', 0):.4f}")
        with col2:
            st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
        with col3:
            st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
        with col4:
            st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")

def generate_deployment_code(selected_model, deployment_framework):
    with st.spinner("🚀 Generating deployment code..."):
        try:
            model_info = st.session_state.trained_models[selected_model]
            
            from core.deployment_generator import DeploymentGenerator
            deployment_gen = DeploymentGenerator()
            deployment_code = deployment_gen.generate_deployment(
                model=model_info['model'],
                model_name=selected_model,
                framework=deployment_framework
            )
            
            st.subheader(f"Deployment Code - {deployment_framework}")
            st.code(deployment_code['code'], language='python')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📥 Download Code",
                    data=deployment_code['code'],
                    file_name=f"{selected_model}_{deployment_framework.lower()}.py",
                    mime="text/python"
                )
            
            with col2:
                if 'dockerfile' in deployment_code:
                    st.download_button(
                        label="🐳 Download Dockerfile",
                        data=deployment_code['dockerfile'],
                        file_name="Dockerfile",
                        mime="text/plain"
                    )
            
            st.success("✅ Deployment code generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Deployment code generation failed: {str(e)}")

if __name__ == "__main__":
    main()
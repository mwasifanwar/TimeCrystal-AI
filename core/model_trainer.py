# core/model_trainer.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import plotly.graph_objects as go

class TimeSeriesTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_models = {}
    
    def train_model(self, data: pd.DataFrame, target_column: str, model_type: str = "Transformer",
                   forecast_horizon: int = 30, epochs: int = 100, learning_rate: float = 0.001,
                   seq_len: int = 100, validation_split: float = 0.2) -> Dict[str, Any]:
        
        processor = TimeSeriesProcessor()
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        values = data[target_column].values
        
        sequences, targets = processor.create_sequences(values, seq_len, forecast_horizon)
        
        train_size = int(len(sequences) * (1 - validation_split))
        train_sequences = sequences[:train_size]
        train_targets = targets[:train_size]
        val_sequences = sequences[train_size:]
        val_targets = targets[train_size:]
        
        train_loader = self._create_data_loader(train_sequences, train_targets, batch_size=32)
        val_loader = self._create_data_loader(val_sequences, val_targets, batch_size=32)
        
        model = self._create_model(model_type, seq_len, forecast_horizon)
        
        lightning_model = TimeSeriesLightningModel(
            model=model,
            learning_rate=learning_rate,
            forecast_horizon=forecast_horizon
        )
        
        trainer = pl.Trainer(
            max_epochs=epochs,
            devices=1 if torch.cuda.is_available() else None,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10, mode="min"),
                ModelCheckpoint(monitor="val_loss", mode="min")
            ],
            enable_progress_bar=True,
            logger=False
        )
        
        trainer.fit(lightning_model, train_loader, val_loader)
        
        val_metrics = trainer.validate(lightning_model, val_loader)[0]
        
        loss_curve = self._plot_loss_curve(lightning_model.training_losses, lightning_model.validation_losses)
        
        return {
            'model': model,
            'metrics': val_metrics,
            'training_time': trainer.fit_loop.epoch_loop.total_time,
            'loss_curve': loss_curve,
            'model_type': model_type,
            'forecast_horizon': forecast_horizon,
            'mse': val_metrics.get('val_mse', 0),
            'mae': val_metrics.get('val_mae', 0),
            'rmse': np.sqrt(val_metrics.get('val_mse', 0))
        }
    
    def train_multiple_models(self, data: pd.DataFrame, target_column: str, 
                            forecast_horizon: int = 30) -> Dict[str, Any]:
        
        models = {}
        
        model_types = ["Transformer", "Informer", "Autoformer"]
        
        for model_type in model_types:
            try:
                model_result = self.train_model(
                    data=data,
                    target_column=target_column,
                    model_type=model_type,
                    forecast_horizon=forecast_horizon,
                    epochs=50
                )
                models[model_type] = model_result
            except Exception as e:
                print(f"Failed to train {model_type}: {str(e)}")
                continue
        
        return models
    
    def _create_model(self, model_type: str, seq_len: int, pred_len: int):
        if model_type == "Transformer":
            return TimeSeriesTransformer(
                d_model=512,
                nhead=8,
                num_layers=6,
                seq_len=seq_len,
                pred_len=pred_len
            )
        elif model_type == "Informer":
            return Informer(
                d_model=512,
                nhead=8,
                num_layers=3,
                seq_len=seq_len,
                pred_len=pred_len
            )
        elif model_type == "Autoformer":
            return Autoformer(
                d_model=512,
                nhead=8,
                num_layers=3,
                seq_len=seq_len,
                pred_len=pred_len
            )
        elif model_type == "Pyraformer":
            return TimeSeriesTransformer(
                d_model=512,
                nhead=8,
                num_layers=4,
                seq_len=seq_len,
                pred_len=pred_len
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_data_loader(self, sequences: np.ndarray, targets: np.ndarray, batch_size: int = 32):
        dataset = TimeSeriesDataset(sequences, targets)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def _plot_loss_curve(self, train_losses: List[float], val_losses: List[float]):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=train_losses,
            mode='lines',
            name='Training Loss',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            y=val_losses,
            mode='lines',
            name='Validation Loss',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Training and Validation Loss',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            height=400
        )
        
        return fig

class TimeSeriesLightningModel(pl.LightningModule):
    def __init__(self, model, learning_rate: float = 0.001, forecast_horizon: int = 30):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.forecast_horizon = forecast_horizon
        self.criterion = nn.MSELoss()
        self.training_losses = []
        self.validation_losses = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        
        self.log("train_loss", loss, prog_bar=True)
        self.training_losses.append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        
        mae = nn.L1Loss()(y_hat, y)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse", loss)
        self.log("val_mae", mae)
        
        self.validation_losses.append(loss.item())
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class AdvancedTimeSeriesTrainer(TimeSeriesTrainer):
    def __init__(self):
        super().__init__()
    
    def train_with_uncertainty(self, data: pd.DataFrame, target_column: str, 
                              model_type: str = "Transformer", forecast_horizon: int = 30,
                              uncertainty_method: str = "monte_carlo", epochs: int = 100) -> Dict[str, Any]:
        
        base_result = self.train_model(
            data=data,
            target_column=target_column,
            model_type=model_type,
            forecast_horizon=forecast_horizon,
            epochs=epochs
        )
        
        quantifier = UncertaintyQuantifier()
        
        values = data[target_column].values
        processor = TimeSeriesProcessor()
        sequences, targets = processor.create_sequences(values, 100, forecast_horizon)
        
        test_sequence = sequences[-1]
        
        uncertainty_result = quantifier.quantify_uncertainty(
            model=base_result['model'],
            data=test_sequence,
            method=uncertainty_method,
            confidence_level=0.95,
            n_simulations=500
        )
        
        base_result['uncertainty'] = uncertainty_result
        base_result['uncertainty_method'] = uncertainty_method
        
        return base_result
    
    def cross_validate(self, data: pd.DataFrame, target_column: str, model_type: str = "Transformer",
                      forecast_horizon: int = 30, n_splits: int = 5) -> Dict[str, Any]:
        
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        values = data[target_column].values
        
        cv_scores = {
            'mse': [],
            'mae': [],
            'rmse': []
        }
        
        for train_idx, test_idx in tscv.split(values):
            train_data = values[train_idx]
            test_data = values[test_idx]
            
            processor = TimeSeriesProcessor()
            train_sequences, train_targets = processor.create_sequences(train_data, 100, forecast_horizon)
            
            if len(train_sequences) == 0:
                continue
            
            model = self._create_model(model_type, 100, forecast_horizon)
            lightning_model = TimeSeriesLightningModel(model=model, forecast_horizon=forecast_horizon)
            
            train_loader = self._create_data_loader(train_sequences, train_targets, batch_size=32)
            
            trainer = pl.Trainer(
                max_epochs=50,
                devices=1 if torch.cuda.is_available() else None,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                enable_progress_bar=False,
                logger=False
            )
            
            trainer.fit(lightning_model, train_loader)
            
            test_sequences, test_targets = processor.create_sequences(test_data, 100, forecast_horizon)
            
            if len(test_sequences) > 0:
                test_loader = self._create_data_loader(test_sequences, test_targets, batch_size=32)
                test_results = trainer.test(lightning_model, test_loader)[0]
                
                cv_scores['mse'].append(test_results.get('test_mse', 0))
                cv_scores['mae'].append(test_results.get('test_mae', 0))
                cv_scores['rmse'].append(np.sqrt(test_results.get('test_mse', 0)))
        
        return {
            'cv_scores': cv_scores,
            'mean_mse': np.mean(cv_scores['mse']),
            'std_mse': np.std(cv_scores['mse']),
            'mean_mae': np.mean(cv_scores['mae']),
            'std_mae': np.std(cv_scores['mae']),
            'mean_rmse': np.mean(cv_scores['rmse']),
            'std_rmse': np.std(cv_scores['rmse']),
            'n_splits': n_splits
        }
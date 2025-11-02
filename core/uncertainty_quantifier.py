# core/uncertainty_quantifier.py
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import scipy.stats as stats

class UncertaintyQuantifier:
    def __init__(self):
        self.methods = {
            'monte_carlo': self.monte_carlo_dropout,
            'conformal': self.conformal_prediction,
            'ensemble': self.ensemble_method,
            'bayesian': self.bayesian_uncertainty
        }
    
    def quantify_uncertainty(self, model, data: np.ndarray, method: str = 'monte_carlo', 
                           confidence_level: float = 0.95, n_simulations: int = 1000,
                           **kwargs) -> Dict[str, Any]:
        
        if method not in self.methods:
            raise ValueError(f"Unsupported uncertainty method: {method}")
        
        uncertainty_func = self.methods[method]
        return uncertainty_func(model, data, confidence_level, n_simulations, **kwargs)
    
    def monte_carlo_dropout(self, model, data: np.ndarray, confidence_level: float = 0.95,
                          n_simulations: int = 1000, **kwargs) -> Dict[str, Any]:
        
        model.train()
        
        input_tensor = torch.FloatTensor(data).unsqueeze(0)
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        predictions = []
        
        for _ in range(n_simulations):
            with torch.no_grad():
                pred = model(input_tensor).cpu().numpy().flatten()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower_bound = mean_prediction - z_score * std_prediction
        upper_bound = mean_prediction + z_score * std_prediction
        
        return {
            'mean': mean_prediction,
            'std': std_prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'predictions': predictions,
            'confidence_level': confidence_level,
            'method': 'monte_carlo_dropout'
        }
    
    def conformal_prediction(self, model, data: np.ndarray, confidence_level: float = 0.95,
                           calibration_data: Tuple[np.ndarray, np.ndarray] = None, 
                           n_simulations: int = 1000, **kwargs) -> Dict[str, Any]:
        
        if calibration_data is None:
            raise ValueError("Calibration data required for conformal prediction")
        
        cal_features, cal_targets = calibration_data
        
        model.eval()
        device = next(model.parameters()).device
        
        cal_residuals = []
        
        with torch.no_grad():
            for i in range(len(cal_features)):
                input_tensor = torch.FloatTensor(cal_features[i]).unsqueeze(0).to(device)
                pred = model(input_tensor).cpu().numpy().flatten()
                residual = np.abs(pred - cal_targets[i])
                cal_residuals.extend(residual)
        
        alpha = 1 - confidence_level
        quantile = np.quantile(cal_residuals, 1 - alpha)
        
        input_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
        with torch.no_grad():
            point_prediction = model(input_tensor).cpu().numpy().flatten()
        
        lower_bound = point_prediction - quantile
        upper_bound = point_prediction + quantile
        
        return {
            'mean': point_prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'quantile': quantile,
            'residuals': cal_residuals,
            'confidence_level': confidence_level,
            'method': 'conformal_prediction'
        }
    
    def ensemble_method(self, model, data: np.ndarray, confidence_level: float = 0.95,
                       n_models: int = 10, n_simulations: int = 1000, **kwargs) -> Dict[str, Any]:
        
        predictions = []
        
        for i in range(n_models):
            model_copy = self._create_model_variant(model)
            
            input_tensor = torch.FloatTensor(data).unsqueeze(0)
            device = next(model.parameters()).device
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                pred = model_copy(input_tensor).cpu().numpy().flatten()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        alpha = 1 - confidence_level
        t_score = stats.t.ppf(1 - alpha/2, df=n_models-1)
        
        lower_bound = mean_prediction - t_score * std_prediction / np.sqrt(n_models)
        upper_bound = mean_prediction + t_score * std_prediction / np.sqrt(n_models)
        
        return {
            'mean': mean_prediction,
            'std': std_prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'predictions': predictions,
            'confidence_level': confidence_level,
            'method': 'ensemble',
            'n_models': n_models
        }
    
    def bayesian_uncertainty(self, model, data: np.ndarray, confidence_level: float = 0.95,
                           n_simulations: int = 1000, **kwargs) -> Dict[str, Any]:
        
        model.train()
        
        input_tensor = torch.FloatTensor(data).unsqueeze(0)
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        predictions = []
        log_vars = []
        
        for _ in range(n_simulations):
            with torch.no_grad():
                pred = model(input_tensor)
                
                if isinstance(pred, tuple):
                    point_pred, log_var = pred
                    point_pred = point_pred.cpu().numpy().flatten()
                    log_var = log_var.cpu().numpy().flatten()
                else:
                    point_pred = pred.cpu().numpy().flatten()
                    log_var = np.zeros_like(point_pred)
                
                predictions.append(point_pred)
                log_vars.append(log_var)
        
        predictions = np.array(predictions)
        log_vars = np.array(log_vars)
        
        mean_prediction = np.mean(predictions, axis=0)
        aleatoric_uncertainty = np.mean(np.exp(log_vars), axis=0)
        epistemic_uncertainty = np.var(predictions, axis=0)
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower_bound = mean_prediction - z_score * np.sqrt(total_uncertainty)
        upper_bound = mean_prediction + z_score * np.sqrt(total_uncertainty)
        
        return {
            'mean': mean_prediction,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'total_uncertainty': total_uncertainty,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'predictions': predictions,
            'confidence_level': confidence_level,
            'method': 'bayesian'
        }
    
    def _create_model_variant(self, model):
        import copy
        
        model_variant = copy.deepcopy(model)
        
        for param in model_variant.parameters():
            noise = torch.randn_like(param) * 0.01
            param.data += noise
        
        return model_variant
    
    def calculate_prediction_intervals(self, predictions: np.ndarray, method: str = 'quantile',
                                     confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        
        if method == 'quantile':
            alpha = 1 - confidence_level
            lower_quantile = alpha / 2
            upper_quantile = 1 - alpha / 2
            
            lower_bound = np.quantile(predictions, lower_quantile, axis=0)
            upper_bound = np.quantile(predictions, upper_quantile, axis=0)
            
        elif method == 'normal':
            mean = np.mean(predictions, axis=0)
            std = np.std(predictions, axis=0)
            
            z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            lower_bound = mean - z_score * std
            upper_bound = mean + z_score * std
        
        else:
            raise ValueError(f"Unsupported interval method: {method}")
        
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'mean': np.mean(predictions, axis=0),
            'method': method,
            'confidence_level': confidence_level
        }
    
    def uncertainty_calibration(self, predictions: np.ndarray, actuals: np.ndarray, 
                              confidence_levels: List[float] = None) -> Dict[str, Any]:
        
        if confidence_levels is None:
            confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        
        calibration_results = {}
        
        for conf_level in confidence_levels:
            intervals = self.calculate_prediction_intervals(predictions, 'quantile', conf_level)
            
            coverage = np.mean((actuals >= intervals['lower_bound']) & (actuals <= intervals['upper_bound']))
            calibration_results[conf_level] = {
                'expected_coverage': conf_level,
                'actual_coverage': coverage,
                'calibration_error': abs(coverage - conf_level)
            }
        
        avg_calibration_error = np.mean([res['calibration_error'] for res in calibration_results.values()])
        
        return {
            'calibration_results': calibration_results,
            'average_calibration_error': avg_calibration_error,
            'well_calibrated': avg_calibration_error < 0.05
        }

class AdvancedUncertaintyQuantifier(UncertaintyQuantifier):
    def __init__(self):
        super().__init__()
        self.methods.update({
            'deep_ensemble': self.deep_ensemble,
            'evidential': self.evidential_uncertainty
        })
    
    def deep_ensemble(self, model, data: np.ndarray, confidence_level: float = 0.95,
                     n_ensembles: int = 5, n_simulations: int = 1000, **kwargs) -> Dict[str, Any]:
        
        ensemble_predictions = []
        
        for ensemble_idx in range(n_ensembles):
            ensemble_model = self._create_ensemble_member(model, ensemble_idx)
            
            member_predictions = []
            for _ in range(n_simulations // n_ensembles):
                input_tensor = torch.FloatTensor(data).unsqueeze(0)
                device = next(model.parameters()).device
                input_tensor = input_tensor.to(device)
                
                with torch.no_grad():
                    pred = ensemble_model(input_tensor).cpu().numpy().flatten()
                    member_predictions.append(pred)
            
            ensemble_predictions.extend(member_predictions)
        
        ensemble_predictions = np.array(ensemble_predictions)
        
        return self.calculate_prediction_intervals(ensemble_predictions, 'quantile', confidence_level)
    
    def evidential_uncertainty(self, model, data: np.ndarray, confidence_level: float = 0.95,
                              n_simulations: int = 1000, **kwargs) -> Dict[str, Any]:
        
        model.eval()
        
        input_tensor = torch.FloatTensor(data).unsqueeze(0)
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            
            if isinstance(output, tuple) and len(output) == 4:
                gamma, nu, alpha, beta = output
                
                gamma = gamma.cpu().numpy().flatten()
                nu = nu.cpu().numpy().flatten()
                alpha = alpha.cpu().numpy().flatten()
                beta = beta.cpu().numpy().flatten()
                
                mean = gamma
                variance = (beta * (1 + nu)) / (nu * (alpha - 1))
                
                aleatoric = beta / (alpha - 1)
                epistemic = variance - aleatoric
                
                std = np.sqrt(variance)
                
                alpha_dist = 1 - confidence_level
                t_val = stats.t.ppf(1 - alpha_dist/2, df=2*alpha)
                
                lower_bound = mean - t_val * std
                upper_bound = mean + t_val * std
                
                return {
                    'mean': mean,
                    'variance': variance,
                    'aleatoric_uncertainty': aleatoric,
                    'epistemic_uncertainty': epistemic,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'evidence_parameters': {
                        'gamma': gamma,
                        'nu': nu,
                        'alpha': alpha,
                        'beta': beta
                    },
                    'confidence_level': confidence_level,
                    'method': 'evidential'
                }
            else:
                return self.monte_carlo_dropout(model, data, confidence_level, n_simulations)
    
    def _create_ensemble_member(self, model, member_idx: int):
        import copy
        import torch.nn as nn
        
        ensemble_member = copy.deepcopy(model)
        
        for name, param in ensemble_member.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        return ensemble_member
    
    def temporal_uncertainty_decomposition(self, predictions: np.ndarray, 
                                         time_horizon: int) -> Dict[str, np.ndarray]:
        
        n_simulations, horizon = predictions.shape
        
        temporal_variance = np.var(predictions, axis=0)
        temporal_mean = np.mean(predictions, axis=0)
        
        uncertainty_growth = np.diff(temporal_variance)
        relative_uncertainty = temporal_variance / (temporal_mean + 1e-8)
        
        return {
            'temporal_variance': temporal_variance,
            'uncertainty_growth': uncertainty_growth,
            'relative_uncertainty': relative_uncertainty,
            'horizon': np.arange(horizon)
        }
"""
Solar Production Prediction Module

This module provides functionality to train XGBoost models for predicting
hourly solar production based on weather data and historical solar output.
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
from typing import Tuple, Optional, Union, Dict, Any
from pathlib import Path

# ML imports
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy import stats

warnings.filterwarnings('ignore')

def solar_performance_metric(rad, ambient, gamma=-0.0035, NOCT=45.0):
    """
    Compute an effective solar performance metric that accounts for irradiance,
    module temperature, and efficiency losses.

    This metric represents an adjusted irradiance value (in W/m²-equivalent)
    that approximates the combined effects of irradiance level, temperature,
    and cell performance characteristics. It is useful for estimating
    photovoltaic (PV) performance under non-standard test conditions.

    Parameters
    ----------
    rad : float or array_like
        Solar irradiance in joules per square centimeter per hour (J/cm²/h).
        Typical range: 0–3600 J/cm²/h (equivalent to 0–1000 W/m²).
    ambient : float or array_like
        Ambient temperature in degrees Celsius (°C).
    gamma : float, optional
        Temperature coefficient of power (per °C). Defaults to -0.0035.
        Negative values indicate power decreases with increasing temperature.
    NOCT : float, optional
        Nominal Operating Cell Temperature in degrees Celsius (°C).
        Defaults to 45.0. Used in estimating cell temperature rise under
        irradiance.

    Returns
    -------
    metric : float or ndarray
        Effective solar performance metric (0–1000 range), representing
        adjusted irradiance in W/m²-equivalent after accounting for
        temperature and irradiance nonlinearity.

    References
    ----------
    - IEC 61853-1: Photovoltaic (PV) module performance testing and energy rating
    - Duffie, J.A., & Beckman, W.A. (2013). *Solar Engineering of Thermal Processes*.

    """
    conv = 10000.0 / 3600.0  # J/cm²/h -> W/m²
    R = rad * conv      # W/m²

    T = ambient + (NOCT - 20.0) / 800.0 * R
    a = 1.0 - 0.2 * np.exp(-R / 200.0)
    irradiance_factor = (R / 1000.0) ** a

    temp_factor = 1.0 + gamma * (T - 25.0)
    temp_factor = np.maximum(temp_factor, 0.0)  # element-wise

    metric = 1000.0 * irradiance_factor * temp_factor
    metric = np.clip(metric, 0.0, 1000.0)       # element-wise clip

    return metric


def create_features(df):
    """
    Perform feature engineering on a time-indexed DataFrame for energy modeling.

    This function generates time-based and weather-related features from
    the input DataFrame, which is assumed to have a DatetimeIndex and
    columns for 'temperature' and 'irradiance'. It also computes derived
    performance metrics for solar energy.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with at least the following columns:
        - 'temperature' : float, ambient temperature in °C
        - 'irradiance' : float, solar irradiance in J/cm²/h
        The DataFrame index must be a pandas.DatetimeIndex.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the following additional columns:
        - 'day_of_week' : int, day of the week (0=Monday, 6=Sunday)
        - 'hour' : int, hour of the day (0–23)
        - 'quarter' : int, quarter of the year (1–4)
        - 'month' : int, month of the year (1–12)
        - 'season' : int, mapped season (0=winter, 1=spring, 2=summer, 3=autumn)
        - 'solar_performance' : float, estimated solar performance metric
    """
    df = df.copy()
    df['day_of_week'] = df.index.dayofweek
    df['hour'] = df.index.hour
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['season'] = df.index.month.map({
        12: 0, 1: 0, 2: 0,    # winter
        3: 1, 4: 1, 5: 1,     # spring
        6: 2, 7: 2, 8: 2,     # summer
        9: 3, 10: 3, 11: 3    # autumn
    })
    df['solar_performance'] = solar_performance_metric(df['irradiance'], df['temperature']).astype(float)
    return df



class SolarPredictor:
    """
    A comprehensive solar production prediction system using XGBoost.
    
    This class handles data preprocessing, outlier detection, feature engineering,
    model training, and prediction for hourly solar production forecasting.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the SolarPredictor.
        
        Args:
            random_state: Random state for reproducible results
        """
        self.random_state = random_state
        self.model = None
        self.feature_columns = [
            'temperature', 'precipitation', 'irradiance',
            'day_of_week', 'hour', 'quarter', 'month', 'season', 'solar_performance'
        ]
        self.is_trained = False
        self.training_stats = {}
        
    def _load_and_process_weather_data(self, weather_data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Load and process weather data.
        
        Args:
            weather_data: Path to CSV file or pandas DataFrame
            
        Returns:
            Processed weather DataFrame with features
        """
        if isinstance(weather_data, str):
            if not os.path.exists(weather_data):
                raise FileNotFoundError(f"Weather data file not found: {weather_data}")
            weather_df = pd.read_csv(weather_data)
        else:
            weather_df = weather_data.copy()
            
        # Ensure datetime column exists
        if 'datetime' in weather_df.columns:
            weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
            weather_df = weather_df.set_index('datetime')
        elif not isinstance(weather_df.index, pd.DatetimeIndex):
            raise ValueError("Weather data must have a 'datetime' column or DatetimeIndex")
            
        # Validate required columns
        required_cols = ['temperature', 'precipitation', 'irradiance']
        missing_cols = [col for col in required_cols if col not in weather_df.columns]
        if missing_cols:
            raise ValueError(f"Weather data missing required columns: {missing_cols}")
            
        # Create features
        return create_features(weather_df)
    
    def _load_and_process_solar_data(self, solar_data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Load and process solar production data.
        
        Args:
            solar_data: Path to CSV file or pandas DataFrame
            
        Returns:
            Processed solar DataFrame
        """
        if isinstance(solar_data, str):
            if not os.path.exists(solar_data):
                raise FileNotFoundError(f"Solar data file not found: {solar_data}")
            solar_df = pd.read_csv(solar_data)
        else:
            solar_df = solar_data.copy()
            
        # Process datetime index
        if 'datetime' in solar_df.columns:
            solar_df['datetime'] = pd.to_datetime(solar_df['datetime'])
            solar_df = solar_df.set_index('datetime')
        elif not isinstance(solar_df.index, pd.DatetimeIndex):
            raise ValueError("Solar data must have a 'datetime' column or DatetimeIndex")
            
        # Ensure solar_kwh column exists
        if 'solar_kwh' not in solar_df.columns:
            raise ValueError("Solar data must contain 'solar_kwh' column")
            
        # Remove negative values
        solar_df = solar_df[solar_df['solar_kwh'] >= 0]
        
        # Resample to hourly if needed
        if len(solar_df) > 0:
            solar_df = solar_df[['solar_kwh']].resample('h').sum()
            
        return solar_df
    
    def _detect_outliers(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive outlier detection for solar production data.
        
        Args:
            merged_data: Merged weather and solar data
            
        Returns:
            Clean data with outliers removed
        """
        print("Detecting outliers...")
        original_size = len(merged_data)
        
        # 1. Context-aware outlier detection by hour
        outlier_mask = pd.Series(False, index=merged_data.index)
        
        for hour in range(24):
            hour_data = merged_data[merged_data['hour'] == hour]
            if len(hour_data) < 10:
                continue
                
            solar_values = hour_data['solar_kwh']
            
            # Statistical outliers (Z-score > 3)
            z_scores = np.abs(stats.zscore(solar_values))
            statistical_outliers = z_scores > 3
            
            # IQR method
            Q1 = solar_values.quantile(0.25)
            Q3 = solar_values.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = (solar_values < (Q1 - 1.5 * IQR)) | (solar_values > (Q3 + 1.5 * IQR))
            
            # Physics-based constraints
            max_expected = {
                **{h: 0.1 for h in [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]},
                6: 0.5, 7: 1.5, 8: 2.5, 9: 3.5,
                10: 4.5, 11: 5.0, 12: 5.5, 13: 5.5, 14: 5.0,
                15: 4.0, 16: 3.0, 17: 2.0, 18: 1.0, 19: 0.5
            }
            physics_outliers = solar_values > max_expected.get(hour, 5.5)
            
            # Combine methods (outlier if flagged by 2+ methods)
            combined_outliers = (statistical_outliers.astype(int) + 
                               iqr_outliers.astype(int) + 
                               physics_outliers.astype(int)) >= 2
            
            outlier_mask.loc[hour_data.index] = combined_outliers
        
        # 2. Seasonal context outlier detection
        seasonal_outlier_mask = pd.Series(False, index=merged_data.index)
        clean_data = merged_data[~outlier_mask]
        
        for season in clean_data['season'].unique():
            for hour in range(6, 20):  # Daylight hours only
                mask = (clean_data['season'] == season) & (clean_data['hour'] == hour)
                season_hour_data = clean_data[mask]
                
                if len(season_hour_data) < 20:
                    continue
                    
                if season_hour_data['irradiance'].std() > 0:
                    irradiance_corr = season_hour_data['solar_kwh'].corr(season_hour_data['irradiance'])
                    
                    if irradiance_corr > 0.5:
                        performance_ratio = season_hour_data['solar_kwh'] / (season_hour_data['solar_performance'] + 1e-6)
                        Q1 = performance_ratio.quantile(0.25)
                        Q3 = performance_ratio.quantile(0.75)
                        IQR = Q3 - Q1
                        
                        ratio_outliers = (performance_ratio < (Q1 - 2.0 * IQR)) | (performance_ratio > (Q3 + 2.0 * IQR))
                        seasonal_outlier_mask.loc[season_hour_data.index] = ratio_outliers
        
        # Apply outlier removal
        final_clean_data = clean_data[~seasonal_outlier_mask]
        
        outliers_removed = original_size - len(final_clean_data)
        print(f"Outliers removed: {outliers_removed} ({outliers_removed/original_size*100:.1f}%)")
        
        return final_clean_data
    
    def train(self, weather_data: Union[str, pd.DataFrame], 
              solar_data: Union[str, pd.DataFrame],
              model_save_path: str,
              test_size: float = 0.2,
              remove_outliers: bool = True,
              tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train the solar prediction model.
        
        Args:
            weather_data: Path to weather CSV or DataFrame with columns:
                         ['datetime', 'temperature', 'precipitation', 'irradiance']
            solar_data: Path to solar CSV or DataFrame with columns:
                       ['datetime', 'solar_kwh']
            model_save_path: Path where to save the trained model
            test_size: Fraction of data to use for testing
            remove_outliers: Whether to apply outlier detection
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training statistics
        """
        print("Starting solar prediction model training...")
        
        # Load and process data
        print("Loading and processing data...")
        weather_features = self._load_and_process_weather_data(weather_data)
        solar_df = self._load_and_process_solar_data(solar_data)
        
        # Merge datasets
        print("Merging weather and solar data...")
        # Align timezone information
        if weather_features.index.tz is None:
            weather_features.index = weather_features.index.tz_localize('UTC', ambiguous='NaT')
        if solar_df.index.tz is None:
            solar_df.index = solar_df.index.tz_localize('UTC', ambiguous='NaT')
            
        weather_features = weather_features.dropna()
        solar_df = solar_df.dropna()
        
        merged_data = weather_features.join(solar_df, how='inner')
        merged_data = merged_data.dropna()
        
        print(f"Merged dataset: {len(merged_data)} records")
        print(f"Date range: {merged_data.index.min()} to {merged_data.index.max()}")
        
        # Outlier detection
        if remove_outliers:
            merged_data = self._detect_outliers(merged_data)
            print(f"Clean dataset: {len(merged_data)} records")
        
        # Prepare features and target
        X = merged_data[self.feature_columns].copy()
        y = merged_data['solar_kwh'].copy()
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Chronological split (important for time series)
        split_idx = int((1 - test_size) * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Model training
        if tune_hyperparameters:
            print("Tuning hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 6],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9]
            }
            
            # Use subset for faster grid search
            subset_size = min(5000, len(X_train))
            X_train_subset = X_train.iloc[:subset_size]
            y_train_subset = y_train.iloc[:subset_size]
            
            grid_search = GridSearchCV(
                estimator=XGBRegressor(random_state=self.random_state, objective='reg:squarederror'),
                param_grid=param_grid,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_subset, y_train_subset)
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
        else:
            # Use default parameters
            best_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8
            }
        
        # Train final model
        print("Training final model...")
        self.model = XGBRegressor(**best_params, random_state=self.random_state, objective='reg:squarederror')
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Store training statistics
        self.training_stats = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'training_samples': len(X_train),
            'testing_samples': len(X_test),
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_)),
            'mean_target': y.mean(),
            'std_target': y.std(),
            'best_params': best_params if tune_hyperparameters else 'default'
        }
        
        # Save model
        os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.', exist_ok=True)
        joblib.dump(self.model, model_save_path)
        self.is_trained = True
        
        print(f"\nModel Training Complete!")
        print(f"Model saved to: {model_save_path}")
        print(f"Training MAE: {train_mae:.4f} kWh")
        print(f"Testing MAE: {test_mae:.4f} kWh")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing R²: {test_r2:.4f}")
        
        return self.training_stats
    
    def predict(self, weather_data: Union[Dict[str, float], pd.DataFrame]) -> Union[float, np.ndarray]:
        """
        Make predictions using the trained model.
        
        Args:
            weather_data: Either a dictionary with single prediction data or DataFrame with multiple predictions
                        Required keys/columns: temperature, precipitation, irradiance,
                                             day_of_week, hour, quarter, month, season
        
        Returns:
            Predicted solar production in kWh
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions. Call train() first.")
        
        if isinstance(weather_data, dict):
            # Single prediction
            # Calculate solar performance metric
            solar_perf = solar_performance_metric(weather_data['irradiance'], weather_data['temperature'])
            
            # Create feature array
            features = np.array([[
                weather_data['temperature'],
                weather_data['precipitation'], 
                weather_data['irradiance'],
                weather_data['day_of_week'],
                weather_data['hour'],
                weather_data['quarter'],
                weather_data['month'],
                weather_data['season'],
                solar_perf
            ]])
            
            prediction = self.model.predict(features)[0]
            return max(0, prediction)  # Ensure non-negative
        
        else:
            # Multiple predictions
            if not isinstance(weather_data, pd.DataFrame):
                raise ValueError("weather_data must be a dictionary or pandas DataFrame")
            
            # Process weather data using the standard method
            weather_data = self._load_and_process_weather_data(weather_data)
            
            # Select required features
            X = weather_data[self.feature_columns]
            predictions = self.model.predict(X)
            return np.maximum(0, predictions)  # Ensure non-negative
    
    @classmethod
    def load_model(cls, model_path: str) -> 'SolarPredictor':
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            SolarPredictor instance with loaded model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        predictor = cls()
        predictor.model = joblib.load(model_path)
        predictor.is_trained = True
        
        return predictor
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        return dict(zip(self.feature_columns, self.model.feature_importances_))
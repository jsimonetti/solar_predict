"""
Solar Prediction Module Usage Examples

This file demonstrates how to use the SolarPredictor module for training
models and making predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Import the solar predictor module
from solar_predictor import SolarPredictor


def example_1_basic_training():
    """
    Example 1: Basic model training using CSV files
    """
    print("="*60)
    print("EXAMPLE 1: BASIC MODEL TRAINING")
    print("="*60)
    
    # Initialize the predictor
    predictor = SolarPredictor(random_state=42)
    
    # Train the model using existing data files
    # Note: These paths are relative to the current working directory
    weather_file = "./development/data_sources/hourly_knmi_weather"  # Your weather CSV file
    
    # For this example, we'll create mock solar data since we need CSV format
    # In real usage, you would point to your actual solar production CSV
    solar_file = "./mock_solar_data.csv"
    create_mock_solar_data(solar_file)  # Helper function to create example data
    
    model_save_path = "./trained_solar_model.pkl"
    
    try:
        # Train the model
        training_stats = predictor.train(
            weather_data=weather_file,
            solar_data=solar_file,
            model_save_path=model_save_path,
            test_size=0.2,
            remove_outliers=True,
            tune_hyperparameters=False  # Set to True for better performance but slower training
        )
        
        print("\nTraining Statistics:")
        print(f"  Training MAE: {training_stats['train_mae']:.4f} kWh")
        print(f"  Testing MAE: {training_stats['test_mae']:.4f} kWh") 
        print(f"  Testing R²: {training_stats['test_r2']:.4f}")
        print(f"  Training samples: {training_stats['training_samples']:,}")
        
        print("\nTop 3 Most Important Features:")
        importance = training_stats['feature_importance']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_features[:3]):
            print(f"  {i+1}. {feature}: {score:.3f}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        return None
    
    return predictor


def example_2_dataframe_training():
    """
    Example 2: Training using pandas DataFrames
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: TRAINING WITH DATAFRAMES")
    print("="*60)
    
    # Create sample weather data
    weather_df = create_sample_weather_data()
    solar_df = create_sample_solar_data()
    
    print(f"Weather data shape: {weather_df.shape}")
    print(f"Solar data shape: {solar_df.shape}")
    print(f"Date range: {weather_df.index.min()} to {weather_df.index.max()}")
    
    # Initialize predictor
    predictor = SolarPredictor(random_state=42)
    
    try:
        # Train using DataFrames
        training_stats = predictor.train(
            weather_data=weather_df,
            solar_data=solar_df,
            model_save_path="./dataframe_model.pkl",
            test_size=0.3,
            remove_outliers=True,
            tune_hyperparameters=False
        )
        
        print(f"\nModel trained successfully!")
        print(f"Test R² Score: {training_stats['test_r2']:.3f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return None
        
    return predictor


def example_3_single_predictions():
    """
    Example 3: Making single predictions
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: SINGLE PREDICTIONS")
    print("="*60)
    
    # Try to load an existing model, or train a new one
    model_path = "./trained_solar_model.pkl"
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        predictor = SolarPredictor.load_model(model_path)
    else:
        print("No existing model found. Training new model...")
        predictor = example_1_basic_training()
        if predictor is None:
            print("Failed to train model. Cannot proceed with predictions.")
            return
    
    # Example prediction for a sunny summer day at noon
    sunny_summer_noon = {
        'temperature': 25.0,      # 25°C
        'precipitation': 0.0,     # No rain
        'irradiance': 300.0,      # High solar irradiance (J/cm²/h)
        'day_of_week': 1,         # Tuesday (0=Monday, 6=Sunday)
        'hour': 12,               # Noon
        'quarter': 2,             # Q2 (April-June)
        'month': 6,               # June
        'season': 2               # Summer (0=winter, 1=spring, 2=summer, 3=autumn)
    }
    
    prediction = predictor.predict(sunny_summer_noon)
    print(f"\nPrediction for sunny summer day at noon: {prediction:.2f} kWh")
    
    # Example prediction for a cloudy winter morning
    cloudy_winter_morning = {
        'temperature': 5.0,       # 5°C
        'precipitation': 2.0,     # Light rain
        'irradiance': 50.0,       # Low irradiance
        'day_of_week': 4,         # Friday
        'hour': 9,                # 9 AM
        'quarter': 1,             # Q1 (Jan-Mar)
        'month': 1,               # January
        'season': 0               # Winter
    }
    
    prediction = predictor.predict(cloudy_winter_morning)
    print(f"Prediction for cloudy winter morning: {prediction:.2f} kWh")
    
    # Example prediction for evening (should be very low)
    evening = {
        'temperature': 15.0,
        'precipitation': 0.0,
        'irradiance': 0.0,        # No sun in evening
        'day_of_week': 2,
        'hour': 20,               # 8 PM
        'quarter': 3,
        'month': 9,
        'season': 3
    }
    
    prediction = predictor.predict(evening)
    print(f"Prediction for evening (8 PM): {prediction:.2f} kWh")


def example_4_batch_predictions():
    """
    Example 4: Making batch predictions with DataFrames
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: BATCH PREDICTIONS")
    print("="*60)
    
    # Load model
    model_path = "./trained_solar_model.pkl"
    
    if os.path.exists(model_path):
        predictor = SolarPredictor.load_model(model_path)
    else:
        print("No model found. Please run example 1 first to train a model.")
        return
    
    # Create sample prediction data for next 24 hours
    prediction_data = create_prediction_sample_data()
    
    print(f"Making predictions for {len(prediction_data)} time points...")
    
    try:
        predictions = predictor.predict(prediction_data)
        
        # Add predictions to the dataframe
        prediction_data['predicted_solar_kwh'] = predictions
        
        print("\nSample predictions:")
        print(prediction_data[['temperature', 'irradiance', 'hour', 'predicted_solar_kwh']].head(10))
        
        # Summary statistics
        total_predicted = prediction_data['predicted_solar_kwh'].sum()
        max_hour = prediction_data.loc[prediction_data['predicted_solar_kwh'].idxmax(), 'hour']
        max_production = prediction_data['predicted_solar_kwh'].max()
        
        print(f"\nPrediction Summary:")
        print(f"  Total predicted production (24h): {total_predicted:.2f} kWh")
        print(f"  Peak production hour: {max_hour}:00")
        print(f"  Peak production: {max_production:.2f} kWh")
        
        # Save predictions
        prediction_data.to_csv('./predictions_sample.csv')
        print(f"\nPredictions saved to: ./predictions_sample.csv")
        
    except Exception as e:
        print(f"Error making predictions: {e}")


def example_5_load_and_predict():
    """
    Example 5: Loading a saved model and making predictions (most common use case)
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: LOAD MODEL AND PREDICT")
    print("="*60)
    
    model_path = "./trained_solar_model.pkl"
    
    try:
        # Load the trained model
        print("Loading trained model...")
        predictor = SolarPredictor.load_model(model_path)
        print("Model loaded successfully!")
        
        # Get feature importance
        importance = predictor.get_feature_importance()
        print("\nFeature Importance:")
        for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {score:.3f}")
        
        # Make a prediction
        current_conditions = {
            'temperature': 20.0,
            'precipitation': 0.0,
            'irradiance': 250.0,
            'day_of_week': 2,  # Wednesday
            'hour': 14,        # 2 PM
            'quarter': 2,
            'month': 5,        # May
            'season': 1        # Spring
        }
        
        prediction = predictor.predict(current_conditions)
        print(f"\nCurrent conditions prediction: {prediction:.2f} kWh")
        
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Please run example 1 first to train a model.")
    except Exception as e:
        print(f"Error: {e}")


# Helper functions to create sample data
def create_mock_solar_data(filename):
    """Create mock solar production data for demonstration"""
    # Read weather data to get the same date range
    try:
        weather_df = pd.read_csv('./development/data_sources/hourly_knmi_weather')
        weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
        
        # Create synthetic solar data based on irradiance and hour
        solar_data = []
        for _, row in weather_df.iterrows():
            hour = row['datetime'].hour
            irradiance = row['irradiance']
            
            # Simple solar production model (for demo purposes)
            base_production = 0
            if 6 <= hour <= 18:  # Daylight hours
                hour_factor = 1 - abs(hour - 12) / 6  # Peak at noon
                irradiance_factor = min(irradiance / 300, 1.0)  # Normalized irradiance
                base_production = hour_factor * irradiance_factor * 4.0  # Max 4 kWh
                
            # Add some random variation
            production = max(0, base_production + np.random.normal(0, 0.1))
            
            solar_data.append({
                'datetime': row['datetime'],
                'solar_kwh': production
            })
        
        solar_df = pd.DataFrame(solar_data)
        solar_df.to_csv(filename, index=False)
        print(f"Created mock solar data: {filename}")
        
    except Exception as e:
        print(f"Error creating mock solar data: {e}")
        # Create minimal sample data
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='h')
        solar_data = []
        for dt in dates:
            hour = dt.hour
            production = max(0, (1 - abs(hour - 12) / 6) * 3.0) if 6 <= hour <= 18 else 0
            production += np.random.normal(0, 0.2)
            production = max(0, production)
            solar_data.append({'datetime': dt, 'solar_kwh': production})
            
        pd.DataFrame(solar_data).to_csv(filename, index=False)


def create_sample_weather_data():
    """Create sample weather data for demonstration"""
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='h')
    
    data = []
    for dt in dates:
        # Simulate seasonal temperature variation
        day_of_year = dt.dayofyear
        base_temp = 15 + 10 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
        
        # Daily temperature variation
        hour_temp_var = 5 * np.sin((dt.hour - 6) * np.pi / 12)
        temperature = base_temp + hour_temp_var + np.random.normal(0, 2)
        
        # Simple irradiance model
        if 6 <= dt.hour <= 18:
            hour_factor = np.sin((dt.hour - 6) * np.pi / 12)
            season_factor = 0.7 + 0.3 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
            irradiance = hour_factor * season_factor * 350 + np.random.normal(0, 50)
            irradiance = max(0, irradiance)
        else:
            irradiance = 0
        
        data.append({
            'datetime': dt,
            'temperature': temperature,
            'precipitation': max(0, np.random.exponential(0.5)),
            'irradiance': irradiance
        })
    
    df = pd.DataFrame(data)
    df = df.set_index('datetime')
    return df


def create_sample_solar_data():
    """Create sample solar data for demonstration"""
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='h')
    
    data = []
    for dt in dates:
        hour = dt.hour
        day_of_year = dt.dayofyear
        
        if 6 <= hour <= 18:
            hour_factor = np.sin((hour - 6) * np.pi / 12)
            season_factor = 0.7 + 0.3 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
            production = hour_factor * season_factor * 4.0 + np.random.normal(0, 0.3)
            production = max(0, production)
        else:
            production = 0
        
        data.append({
            'datetime': dt,
            'solar_kwh': production
        })
    
    df = pd.DataFrame(data)
    df = df.set_index('datetime')
    return df


def create_prediction_sample_data():
    """Create sample data for making predictions"""
    # Next 24 hours starting from now
    start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    dates = pd.date_range(start=start_time, periods=24, freq='h')
    
    data = []
    for dt in dates:
        # Sample weather conditions for next 24 hours
        data.append({
            'temperature': 15 + 5 * np.sin((dt.hour - 6) * np.pi / 12) + np.random.normal(0, 1),
            'precipitation': max(0, np.random.exponential(0.3)),
            'irradiance': max(0, 300 * np.sin(max(0, dt.hour - 6) * np.pi / 12)) if 6 <= dt.hour <= 18 else 0,
            'day_of_week': dt.dayofweek,
            'hour': dt.hour,
            'quarter': (dt.month - 1) // 3 + 1,
            'month': dt.month,
            'season': (dt.month % 12 // 3)
        })
    
    return pd.DataFrame(data, index=dates)


if __name__ == "__main__":
    """
    Run all examples
    """
    print("SOLAR PREDICTION MODULE EXAMPLES")
    print("=" * 60)
    print("This script demonstrates various ways to use the SolarPredictor module.")
    print()
    
    try:
        # Example 1: Basic training
        predictor = example_1_basic_training()
        
        # Example 2: DataFrame training (only if first example worked)
        if predictor is not None:
            example_2_dataframe_training()
        
        # Example 3: Single predictions
        example_3_single_predictions()
        
        # Example 4: Batch predictions
        example_4_batch_predictions()
        
        # Example 5: Load and predict (most common usage)
        example_5_load_and_predict()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED!")
        print("="*60)
        print("\nKey files created:")
        print("  - trained_solar_model.pkl (trained model)")
        print("  - mock_solar_data.csv (sample solar data)")
        print("  - predictions_sample.csv (sample predictions)")
        print("\nYou can now use these files to make predictions in your own code!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
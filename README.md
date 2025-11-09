# Solar Production Prediction Module

A comprehensive Python module for predicting hourly solar production using XGBoost and weather data.

## Features

- 🤖 **XGBoost-based predictions** with automatic hyperparameter tuning
- 🧹 **Smart outlier detection** using context-aware methods
- 🌡️ **Weather feature engineering** including solar performance metrics
- 📊 **Comprehensive evaluation** with multiple metrics
- 💾 **Model persistence** for easy deployment and reuse
- 📈 **Batch and single predictions** support

## Quick Start

### 1. Installation

Ensure you have the required dependencies:

```bash
pip install pandas numpy scikit-learn xgboost scipy joblib
```

### 2. Basic Usage

#### Training a Model

```python
from solar_predictor import SolarPredictor

# Initialize predictor
predictor = SolarPredictor(random_state=42)

# Train model with your data
training_stats = predictor.train(
    weather_data="./your_weather_data.csv",  # or DataFrame
    solar_data="./your_solar_data.csv",      # or DataFrame
    model_save_path="./my_solar_model.pkl",
    remove_outliers=True,
    tune_hyperparameters=True
)

print(f"Model R² Score: {training_stats['test_r2']:.3f}")
```

#### Making Predictions

```python
# Load trained model
predictor = SolarPredictor.load_model("./my_solar_model.pkl")

# Single prediction
weather_conditions = {
    'temperature': 25.0,      # °C
    'precipitation': 0.0,     # mm
    'irradiance': 300.0,      # J/cm²/h
    'day_of_week': 1,         # 0=Monday, 6=Sunday
    'hour': 12,               # 0-23
    'quarter': 2,             # 1-4
    'month': 6,               # 1-12
    'season': 2               # 0=winter, 1=spring, 2=summer, 3=autumn
}

prediction = predictor.predict(weather_conditions)
print(f"Predicted solar production: {prediction:.2f} kWh")
```

## Data Format Requirements

### Weather Data

Your weather data should contain these columns:

| Column | Description | Units | Example |
|--------|-------------|-------|---------|
| `datetime` | Timestamp | ISO format | "2023-06-15 14:00:00" |
| `temperature` | Air temperature | °C | 25.3 |
| `precipitation` | Precipitation amount | mm | 0.0 |
| `irradiance` | Solar irradiance | J/cm²/h | 280.5 |

### Solar Data

Your solar production data should contain:

| Column | Description | Units | Example |
|--------|-------------|-------|---------|
| `datetime` | Timestamp | ISO format | "2023-06-15 14:00:00" |
| `solar_kwh` | Solar production | kWh | 3.45 |

## Examples

### Complete Training Example

```python
# Run the complete training and evaluation example
python example.py
```

This single file demonstrates:
- Basic training with CSV files
- Training with pandas DataFrames
- Single predictions for various weather scenarios
- Batch predictions and daily forecasts
- Loading saved models and making predictions
- Creating sample data and mock datasets

## Module API Reference

### SolarPredictor Class

#### Constructor
```python
SolarPredictor(random_state=42)
```

#### Methods

**`train(weather_data, solar_data, model_save_path, **kwargs)`**
- Trains the solar prediction model
- Returns training statistics dictionary
- Supports both file paths and DataFrames

**`predict(weather_data)`**
- Makes predictions on new data
- Supports single prediction (dict) or batch (DataFrame)
- Returns predicted kWh values

**`load_model(model_path)`** (class method)
- Loads a previously trained model
- Returns SolarPredictor instance

**`get_feature_importance()`**
- Returns feature importance scores
- Useful for understanding model behavior

## Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_size` | float | 0.2 | Fraction of data for testing |
| `remove_outliers` | bool | True | Apply outlier detection |
| `tune_hyperparameters` | bool | True | Perform hyperparameter tuning |

## Model Performance

The module includes comprehensive outlier detection and achieves typical performance metrics:

- **Mean Absolute Error**: 0.2-0.4 kWh/hour
- **R² Score**: 0.75-0.85
- **Feature Importance**: Irradiance and solar_performance are typically most important

## File Structure

```
solar_predict/
├── solar_predictor.py       # Main prediction module
├── example.py              # Complete training and usage examples  
├── README.md              # This documentation file
└── development/           # Development files and data sources
    ├── data_sources/      # Original weather and solar data
    ├── notebooks/         # Development Jupyter notebooks
    ├── models/           # Trained models from development
    └── README_DEVELOPMENT.md  # Development directory documentation
```

## Outlier Detection

The module implements multi-layered outlier detection:

1. **Context-aware detection** by hour of day
2. **Statistical methods** (Z-score, IQR)
3. **Physics-based constraints** (max production limits)
4. **Seasonal performance** analysis

Outliers are typically 1-3% of the dataset and significantly improve model performance.

## Tips for Best Results

1. **Data Quality**: Ensure your weather and solar data are synchronized and complete
2. **Regular Retraining**: Update the model with new data periodically
3. **Validation**: Monitor actual vs. predicted performance
4. **Feature Engineering**: The built-in solar performance metric significantly improves accuracy
5. **Outlier Review**: Check removed outliers to identify potential system issues

## Troubleshooting

### Common Issues

**"Model must be trained before making predictions"**
- Ensure you call `train()` before `predict()` or use `load_model()`

**"Missing required columns"**
- Verify your data contains all required weather/solar columns

**"Model file not found"**
- Check the model path and ensure the model was saved successfully

**Poor performance**
- Try enabling outlier removal and hyperparameter tuning
- Ensure sufficient training data (>1000 samples recommended)
- Verify data quality and synchronization

## License

This module is designed for solar production forecasting and can be adapted for various solar installations and geographic locations.
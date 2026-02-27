# Used Car Price Prediction

A machine learning pipeline for predicting used car prices in the Saudi market using LightGBM, Random Forest, and CatBoost.

## Project Overview

This project was developed at **King Khalid University** and builds a complete ML pipeline that:

- Loads and explores a used car dataset (`cars.csv`)
- Cleans and engineers features
- Compares three models: **LightGBM**, **Random Forest**, and **CatBoost**
- Performs detailed error analysis by brand, year, and price range
- Provides SHAP-based model explainability
- Exports trained models and configuration for deployment

## Features

- **12 input features**: Make, Type, Origin, Color, Options, Fuel Type, Gear Type, Region, Year, Engine Size, Mileage, Car Age
- **Target**: Price (SAR)
- **Best Model**: LightGBM with early stopping
- **Explainability**: SHAP values for individual predictions
- **Confidence Intervals**: 90% prediction intervals

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

1. Place your `cars.csv` dataset in the project directory
2. Run the pipeline:

```bash
python modle.py
```

3. Outputs:
   - `car_model.pkl` - Trained LightGBM model
   - `encoder.pkl` - Feature encoder
   - `bundle.json` - Configuration and dropdown options
   - `metrics.json` - Performance metrics
   - Various `.png` charts for analysis

## Project Structure

```
my-python-project/
├── modle.py           # Main ML pipeline
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── .gitignore         # Git ignore rules
```

## License

This project is for educational purposes at King Khalid University.
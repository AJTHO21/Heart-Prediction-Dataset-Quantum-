# Heart Disease Prediction Analysis

This project analyzes a dataset containing both traditional medical indicators and a quantum-inspired feature for heart disease prediction. The dataset includes 500 samples with features such as Age, Gender, Blood Pressure, Cholesterol, Heart Rate, and a unique QuantumPatternFeature.

## Dataset Description

The dataset combines traditional medical indicators with a quantum-inspired feature to predict heart disease. Features include:
- Age (continuous)
- Gender (binary: 0/1)
- Blood Pressure (continuous)
- Cholesterol (continuous)
- Heart Rate (continuous)
- QuantumPatternFeature (continuous)
- Heart Disease (target: 0/1)

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis:
   ```bash
   python src/data_analysis.py
   ```

## Project Structure

- `src/data_analysis.py`: Main analysis script
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation

## Analysis

The analysis includes:
1. Exploratory Data Analysis (EDA)
2. Feature importance analysis
3. Model development and evaluation
4. Visualization of results 
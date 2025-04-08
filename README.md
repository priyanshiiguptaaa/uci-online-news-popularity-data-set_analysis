# News Popularity Regularization Analysis

## Project Overview

This project demonstrates the application and comparison of different regularization techniques (L1 and L2) on the UCI Online News Popularity dataset. The goal is to predict article popularity (measured by shares) while understanding how different regularization methods affect model coefficients and performance.

## Dataset

The UCI Online News Popularity dataset contains features about Mashable articles and their social media popularity. The dataset includes:
- 39,644 articles
- 61 features (including the target variable)
- Target variable: number of social media shares

Key features include article metadata, keyword metrics, natural language processing metrics, and timing information.

## Implementation

The implementation compares three regression models:
1. Linear Regression (No Regularization)
2. Ridge Regression (L2 Regularization)
3. Lasso Regression (L1 Regularization)

The code:
- Preprocesses data by removing non-predictive columns
- Scales features using StandardScaler
- Trains the three regression models
- Evaluates model performance with MSE and R² metrics
- Analyzes coefficient distributions and feature importance
- Visualizes the differences between regularization techniques

## Key Visualizations

The project includes several visualizations:
- Histograms of coefficient distributions across models
- Stem plots showing individual coefficient values
- Density plots (KDE) for better distribution comparison
- Separate density plots for each model to avoid scale issues
- Feature importance rankings

## Results

Results highlight how:
- Lasso regression performs feature selection by setting many coefficients to exactly zero
- Ridge regression shrinks coefficients toward zero but rarely sets them exactly to zero
- Different features are prioritized by different regularization techniques
- Each model's performance metrics illustrate the bias-variance tradeoff

## Usage

1. Ensure required packages are installed:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Run the script:
```python
python news_regularization_analysis.py
```

3. Review the generated visualizations and model performance metrics.

## Technical Notes

- The dataset has column names with leading spaces that need to be stripped
- Feature scaling is essential for proper regularization behavior
- Hyperparameters (alpha values) can be tuned for optimal performance

## Further Improvements

Potential enhancements include:
- Implementing cross-validation for hyperparameter tuning
- Testing additional regularization techniques (e.g., ElasticNet)
- Applying feature engineering to improve prediction accuracy
- Using more advanced metrics beyond MSE and R²
- Adding non-linear models for comparison

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
print("Loading the dataset...")
df = pd.read_csv("/kaggle/input/uci-online-news-popularity-data-set/OnlineNewsPopularity.csv")

# Display basic information about the dataset
print(f"\nDataset shape: {df.shape}")
print("\nFirst 5 rows of the dataset:")
print(df.head())

# From the output we can see that the actual column name includes a space
# The target column is ' shares' with a space prefix, not 'shares'
target_column = ' shares'  # Note the space before 'shares'

print(f"\nBasic statistics of the target variable '{target_column}':")
print(df[target_column].describe())

# Data preprocessing
# Fix column names by stripping leading/trailing spaces
df.columns = df.columns.str.strip()
print("\nColumn names after stripping spaces:")
print(df.columns.tolist())

# Now the target column is 'shares' without the space
target_column = 'shares'

# Remove non-predictive columns
print("\nRemoving non-predictive columns...")
# Dropping URL and timedelta columns as they're not useful for prediction
df = df.drop(['url', 'timedelta'], axis=1)

# Check for missing values
print(f"\nMissing values in the dataset: {df.isnull().sum().sum()}")

# Separate features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

print(f"\nNumber of features: {X.shape[1]}")
print(f"Some feature names: {list(X.columns)[:5]}")

# Split the data into training and testing sets
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print some stats about scaled data
print(f"Mean of first 5 scaled features: {np.mean(X_train_scaled[:, :5], axis=0)}")
print(f"Standard deviation of first 5 scaled features: {np.std(X_train_scaled[:, :5], axis=0)}")

# Function to train and evaluate models
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Print metrics
    print(f"\n{model_name} Results:")
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Testing MSE: {test_mse:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    
    # Print some coefficient statistics
    if hasattr(model, 'coef_'):
        print(f"Number of features used: {np.sum(model.coef_ != 0)}")
        print(f"Max coefficient value: {np.max(model.coef_):.4f}")
        print(f"Min coefficient value: {np.min(model.coef_):.4f}")
    
    return model.coef_ if hasattr(model, 'coef_') else None

# Train models with different regularization techniques
print("\nTraining Linear Regression (No Regularization)...")
lr = LinearRegression()
lr_coef = train_and_evaluate_model(lr, X_train_scaled, y_train, X_test_scaled, y_test, "Linear Regression")

print("\nTraining Ridge Regression (L2 Regularization)...")
ridge = Ridge(alpha=10.0, random_state=42)
ridge_coef = train_and_evaluate_model(ridge, X_train_scaled, y_train, X_test_scaled, y_test, "Ridge Regression")

print("\nTraining Lasso Regression (L1 Regularization)...")
lasso = Lasso(alpha=1.0, random_state=42)
lasso_coef = train_and_evaluate_model(lasso, X_train_scaled, y_train, X_test_scaled, y_test, "Lasso Regression")

# Print a random sample of coefficients for comparison
random_features = np.random.choice(range(X.shape[1]), 5, replace=False)
print("\nRandom sample of coefficients for comparison:")
for i, feature_idx in enumerate(random_features):
    feature_name = X.columns[feature_idx]
    print(f"{feature_name[:30]:30} - LR: {lr_coef[feature_idx]:10.4f}, Ridge: {ridge_coef[feature_idx]:10.4f}, Lasso: {lasso_coef[feature_idx]:10.4f}")

# Visualize coefficient distributions
plt.figure(figsize=(12, 6))
plt.hist(lr_coef, bins=50, alpha=0.5, label='Linear (No Reg)', color='yellow')
plt.hist(ridge_coef, bins=50, alpha=0.5, label='Ridge (L2)', color='red')
plt.hist(lasso_coef, bins=50, alpha=0.5, label='Lasso (L1)', color='blue')
plt.xlabel('Coefficient Value')
plt.ylabel('Frequency')
plt.title('Coefficient Distribution Comparison')
plt.legend()
plt.grid(True)
plt.savefig('coef_histogram.png')
plt.show()

# Visualize coefficient values with a better visualization
plt.figure(figsize=(14, 8))
plt.subplot(3, 1, 1)
plt.stem(lr_coef, linefmt='y-', markerfmt='yo', basefmt='k-')
plt.title('Linear Regression Coefficients')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(ridge_coef, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.title('Ridge Regression Coefficients')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.stem(lasso_coef, linefmt='b-', markerfmt='bo', basefmt='k-')
plt.title('Lasso Regression Coefficients')
plt.xlabel('Feature Index')
plt.grid(True)

plt.tight_layout()
plt.savefig('coef_comparison.png')
plt.show()

# Visualize coefficient density plots - better than the simple histogram
plt.figure(figsize=(12, 6))
sns.kdeplot(lr_coef, label='Linear (No Reg)', color='yellow')
sns.kdeplot(ridge_coef, label='Ridge (L2)', color='red')
sns.kdeplot(lasso_coef, label='Lasso (L1)', color='blue')
plt.xlabel('Coefficient Value')
plt.ylabel('Density')
plt.title('Coefficient Distribution Comparison')
plt.legend()
plt.grid(True)
plt.savefig('coef_density.png')
plt.show()

# For a better visualization of the coefficient distribution differences
# (This should fix the extreme scales issue you were seeing)
plt.figure(figsize=(15, 5))

# Add a subplot for each model with appropriate scale
plt.subplot(1, 3, 1)
sns.kdeplot(lr_coef, color='yellow')
plt.title('Linear Regression Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Density')
plt.grid(True)

plt.subplot(1, 3, 2)
sns.kdeplot(ridge_coef, color='red')
plt.title('Ridge (L2) Coefficients')
plt.xlabel('Coefficient Value')
plt.grid(True)

plt.subplot(1, 3, 3)
sns.kdeplot(lasso_coef, color='blue')
plt.title('Lasso (L1) Coefficients')
plt.xlabel('Coefficient Value')
plt.grid(True)

plt.tight_layout()
plt.savefig('coef_density_separate.png')
plt.show()

# Feature importance analysis
# Get top 10 features by absolute coefficient value for each model
def get_top_features(coef, feature_names, n=10):
    # Get the indices of the top n coefficients by absolute value
    top_indices = np.argsort(np.abs(coef))[-n:]
    # Return feature names and coefficient values
    return [(feature_names[i], coef[i]) for i in reversed(top_indices)]

print("\nTop 10 features by importance (Linear Regression):")
for feature, coef in get_top_features(lr_coef, X.columns, 10):
    print(f"{feature[:30]:30}: {coef:.4f}")

print("\nTop 10 features by importance (Ridge Regression):")
for feature, coef in get_top_features(ridge_coef, X.columns, 10):
    print(f"{feature[:30]:30}: {coef:.4f}")

print("\nTop 10 features by importance (Lasso Regression):")
for feature, coef in get_top_features(lasso_coef, X.columns, 10):
    print(f"{feature[:30]:30}: {coef:.4f}")

# Count number of zero coefficients in each model
print("\nNumber of zero coefficients (feature selection):")
print(f"Linear Regression: {np.sum(np.abs(lr_coef) < 1e-10)}")
print(f"Ridge Regression: {np.sum(np.abs(ridge_coef) < 1e-10)}")
print(f"Lasso Regression: {np.sum(np.abs(lasso_coef) < 1e-10)}")

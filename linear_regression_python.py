import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------
# Script Name: linear_regression_python.py
# Author: Kehinde Soetan
# Date: 2025-06-30
# Description: Command-line linear regression script.
# Usage: python linear_regression_python.py <filename> <x_column> <y_column>
# ---------------------------------------------

# Command-line argument check
if len(sys.argv) != 4:
    print("Usage: python linear_regression_python.py <filename> <x_column> <y_column>")
    sys.exit(1)

filename = sys.argv[1]
x_column = sys.argv[2]
y_column = sys.argv[3]

# Load dataset
df = pd.read_csv(filename)

# Check columns exist
if x_column not in df.columns or y_column not in df.columns:
    print(f"Error: One or both columns '{x_column}' and '{y_column}' not found in dataset.")
    sys.exit(1)

# Select columns and drop missing data
df = df[[x_column, y_column]].dropna()

X = df[[x_column]]
y = df[y_column]

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Calculate metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Print evaluation
print(f"\nModel Evaluation:")
print(f"  R² Score: {r2:.4f}")
print(f"  Mean Squared Error: {mse:.4f}")
print(f"  Slope (Coefficient): {model.coef_[0]:.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

# Plot results
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.title(f'{x_column} vs {y_column}')
plt.legend()
plt.tight_layout()
plt.savefig('linear_regression_python_output.png')
plt.close()

print("\nPlot saved as 'linear_regression_python_output.png'")

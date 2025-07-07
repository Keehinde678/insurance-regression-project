import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Check for the correct number of arguments
if len(sys.argv) != 4:
    print("Usage: python linear_regression_python.py <filename> <x_column> <y_column>")
    sys.exit(1)

filename = sys.argv[1]
x_column = sys.argv[2]
y_column = sys.argv[3]

# Load the dataset
df = pd.read_csv(filename)

# Check if columns exist
if x_column not in df.columns or y_column not in df.columns:
    print(f"Error: Columns '{x_column}' or '{y_column}' not found in the dataset.")
    sys.exit(1)

# Prepare the data
X = df[[x_column]].dropna()
y = df[y_column].dropna()

# Align indices after dropping NaNs to avoid mismatch
X, y = X.align(y, join='inner')

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate evaluation metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Print results
print(f"\nModel Evaluation:")
print(f"  R² Score: {r2:.4f}")
print(f"  Mean Squared Error: {mse:.4f}")
print(f"  Slope (Coefficient): {model.coef_[0]:.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

# Plot the regression
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.title(f'Linear Regression: {x_column} vs {y_column}')
plt.legend()
plt.tight_layout()
plt.savefig('linear_regression_python_output.png')
plt.close()

print("\nPlot saved as 'linear_regression_python_output.png'")

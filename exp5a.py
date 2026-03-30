# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset (Example Data)
data = {
    'square_feet': [1500, 1800, 2400, 3000, 3500, 4000, 4500, 5000, 5500, 6000],
    'num_bedrooms': [3, 4, 3, 5, 4, 6, 5, 7, 6, 8],
    'num_bathrooms': [2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
    'year_built': [2000, 2005, 2010, 2015, 2020, 1995, 1985, 2018, 2012, 2008],
    'price': [300000, 350000, 450000, 500000, 600000, 650000, 700000, 800000, 850000, 900000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print("Dataset Sample:\n", df.head())

# Select independent variables (features) and dependent variable (target)
X = df[['square_feet', 'num_bedrooms', 'num_bathrooms', 'year_built']]  # Features
y = df['price']  # Target variable

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict house prices for the test set
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print model coefficients
print("\nModel Coefficients:")
print("Intercept:", model.intercept_)
print("Feature Coefficients:", model.coef_)

# Print performance metrics
print("\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Visualization: Actual vs Predicted Prices
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
# Experience (in years) and corresponding Salary (in $1000s)
data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [30, 35, 40, 50, 55, 60, 65, 75, 80, 95]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Splitting dataset into features (X) and target variable (y)
X = df[['Experience']]
y = df['Salary']

# Splitting into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Visualizing the results
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Testing Data')

# Regression line
plt.plot(X, model.predict(X), color='green', linewidth=2, label='Regression Line')

plt.xlabel('Experience (Years)')
plt.ylabel('Salary ($1000s)')
plt.title('Experience vs Salary')
plt.legend()
plt.show()
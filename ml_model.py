import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data (replace with your own dataset)
np.random.seed(42)
n_samples = 1000
n_features = 5

X = np.random.randn(n_samples, n_features)
# True coefficients
true_coef = np.array([0.5, 2.0, -1.0, 3.0, 0.8])
# Generate target variable with some noise
y = np.dot(X, true_coef) + np.random.normal(0, 0.1, n_samples)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')
print(f'R² Score: {r2:.4f}')

# Print feature coefficients
for i, coef in enumerate(model.coef_):
    print(f'Feature {i+1} coefficient: {coef:.4f}')

# Example prediction for new data
new_data = np.random.randn(1, n_features)
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print(f'\nPrediction for new data: {prediction[0]:.4f}')

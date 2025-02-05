import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filepath, target_column):
    """Load and preprocess data from CSV file"""
    
    # Read CSV file
    df = pd.read_csv(filepath)
    
    # Convert date column to datetime, was giving issues cuz string
    date_cols = df.select_dtypes(include=['object']).columns
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])

            # convert string dates to meaningful features
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_year'] = df[col].dt.year
            
            # Drop original date column
            df = df.drop(columns=[col])
        except:
            continue
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y

def train_model(X, y):
    """Train linear regression model"""
    # Convert any remaining string columns to numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Remove any columns with NaN values
    X = X.dropna(axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, mse, r2, X_train.columns

def main():
    filepath = "load_files/merged_load_data_verticalFormat_cleaned.csv"  # Replace with your CSV file path
    target_column = "BasicLoad"  # Your target column
    
    # Load data
    X, y = load_data(filepath, target_column)
    
    # Train and evaluate model
    model, scaler, mse, r2, feature_names = train_model(X, y)
    
    # Print results
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'R² Score: {r2:.4f}')
    
    # Print feature coefficients
    for name, coef in zip(feature_names, model.coef_):
        print(f'{name} coefficient: {coef:.4f}')

if __name__ == "__main__":
    main()

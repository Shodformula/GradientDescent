import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
url = 'https://raw.githubusercontent.com/Shodformula/Linear-Regression-Gradient-Descent/main/AirQualityUCI.csv'
df = pd.read_csv(url, sep=';', on_bad_lines='skip')

# Pre-processing
# Replace commas with dots in numeric columns only
numeric_cols = df.columns.drop(['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'])
df[numeric_cols] = df[numeric_cols].replace(',', '.', regex=True).astype(float)

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16'])

# Drop any rows with missing values
df = df.dropna()

# Features (Temperature, Humidity, NOx) and target (CO)
X = df[['CO(GT)', 'RH', 'NOx(GT)']]
y = df['T']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define hyperparameters to tune
learning_rates = [0.001, 0.01, 0.1]
max_iters = [100, 500, 1000]

# Open a log file to store the results
log_file = open('hyperparameter_log.txt', 'w')
log_file.write("Learning Rate, Max Iterations, Train MSE, Test MSE, R² Score, Weights\n")

results = []

for lr in learning_rates:
    for n_iter in max_iters:
        # Create a new model with each combination of hyperparameters
        model = SGDRegressor(learning_rate='constant', eta0=lr, max_iter=n_iter, tol=1e-3)

        # Train the model
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate MSE for both train and test sets
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        # Calculate R² score
        r2 = r2_score(y_test, y_test_pred)
        
        # Get the model's weight coefficients
        weights = model.coef_

        # Save results for analysis
        results.append({
            'learning_rate': lr,
            'max_iter': n_iter,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'r2_score': r2,
            'weights': weights
        })

        # Log the hyperparameters and results to the log file
        log_file.write(f"{lr}, {n_iter}, {train_mse:.4f}, {test_mse:.4f}, {r2:.4f}, {weights.tolist()}\n")

        # Print the results
        print(f"Learning Rate: {lr}, Max Iterations: {n_iter}, Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}, R²: {r2:.4f}, Weights: {weights}")

# Close the log file
log_file.close()

# Plot the results
learning_rate_values = [result['learning_rate'] for result in results]
iteration_values = [result['max_iter'] for result in results]
test_mse_values = [result['test_mse'] for result in results]

plt.figure(figsize=(10, 6))
plt.scatter(learning_rate_values, test_mse_values, c=iteration_values, cmap='viridis')
plt.colorbar(label='Max Iterations')
plt.xlabel('Learning Rate')
plt.ylabel('Test MSE')
plt.title('Test MSE for Different Learning Rates and Iterations')
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
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
average_options = [True, False]
stop_values = [1e-3, 1e-4]  # Replaced tol with stop

# Open a log file to store the results
log_file = open('hyperparameter_log.txt', 'w')
log_file.write("Learning Rate, Max Iterations, Average, Stop, Train MSE, Test MSE, R² Score, Explained Variance, Weights\n")

results = []

# Train models and evaluate performance with different hyperparameters
for lr in learning_rates:
    for n_iter in max_iters:
        for avg in average_options:
            for stop in stop_values:
                model = SGDRegressor(learning_rate='constant', eta0=lr, max_iter=n_iter, tol=stop, average=avg)  # Stop replaces tol
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                r2 = r2_score(y_test, y_test_pred)
                explained_variance = explained_variance_score(y_test, y_test_pred)
                weights = model.coef_

                # Save results for analysis
                results.append({
                    'learning_rate': lr,
                    'max_iter': n_iter,
                    'average': avg,
                    'stop': stop,  # Changed tol to stop
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'r2_score': r2,
                    'explained_variance': explained_variance,
                    'weights': weights
                })

                # Log the hyperparameters and results to the log file (without printing to console)
                log_file.write(f"{lr}, {n_iter}, {avg}, {stop}, {train_mse:.4f}, {test_mse:.4f}, {r2:.4f}, {explained_variance:.4f}, {weights.tolist()}\n")

# Close the log file
log_file.close()

# Find the best model (lowest Test MSE)
best_result = min(results, key=lambda x: x['test_mse'])

# Output evaluation statistics for the best model
print("\n|=== Best Model Results ===|")
print(f"Model Test MSE: {best_result['test_mse']:.4f}")
print(f"Model R²: {best_result['r2_score']:.4f}")
print(f"Model Explained Variance: {best_result['explained_variance']:.4f}")
print(f"Model Weights (Coefficients): {best_result['weights']}")
print("|=== Hyperparameters ===|")
print(f"Learning Rate: {best_result['learning_rate']}")
print(f"Max Iterations: {best_result['max_iter']}")
print(f"Average: {best_result['average']}")
print(f"Stop Value: {best_result['stop']}")  # Changed tol to stop

# Plot Test MSE vs Learning Rate and Max Iterations
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

# Plot Predicted vs Actual values for the best model
best_model = SGDRegressor(learning_rate='constant', eta0=best_result['learning_rate'], max_iter=best_result['max_iter'], tol=best_result['stop'], average=best_result['average'])  # Stop replaces tol
best_model.fit(X_train, y_train)
y_test_pred_best = best_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred_best, alpha=0.7)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values (Best Model)")
plt.show()

# New plot: MSE vs Number of Iterations for Different Learning Rates
plt.figure(figsize=(10, 6))
for lr in learning_rates:
    mse_for_lr = [result['test_mse'] for result in results if result['learning_rate'] == lr]
    iters_for_lr = [result['max_iter'] for result in results if result['learning_rate'] == lr]
    plt.plot(iters_for_lr, mse_for_lr, label=f'Learning Rate = {lr}')
plt.xlabel('Number of Iterations')
plt.ylabel('Test MSE')
plt.title('Test MSE vs Number of Iterations for Different Learning Rates')
plt.legend()
plt.show()

# New plot: Target Variable (Temperature, T) vs CO(GT)
plt.figure(figsize=(10, 6))
plt.scatter(df['CO(GT)'], y, alpha=0.5)
plt.xlabel("CO(GT)")
plt.ylabel("Temperature (T)")
plt.title("Temperature vs CO(GT)")
plt.show()

# New plot: Target Variable (Temperature, T) vs NOx(GT)
plt.figure(figsize=(10, 6))
plt.scatter(df['NOx(GT)'], y, alpha=0.5, color='green')
plt.xlabel("NOx(GT)")
plt.ylabel("Temperature (T)")
plt.title("Temperature vs NOx(GT)")
plt.show()

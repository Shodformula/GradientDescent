import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/Shodformula/Linear-Regression-Gradient-Descent/main/AirQualityUCI.csv'
df = pd.read_csv(url, sep=';', on_bad_lines='skip')

# Pre-processing
numeric_cols = df.columns.drop(['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'])
df[numeric_cols] = df[numeric_cols].replace(',', '.', regex=True).astype(float)
df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16'])
df = df.dropna()

# Features, Target is temperature
X = df[['CO(GT)', 'RH', 'NOx(GT)']]
y = df['T']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameters
learning_rates = [0.001, 0.01, 0.1]
max_iters = [100, 500, 1000]
average_options = [True, False]
stop_values = [1e-3, 1e-4]

# Log results
log_file = open('hyper_log.txt', 'w')
log_file.write("Learning Rate, Max Iterations, Average, Stop, Train MSE, Test MSE, R² Score, Explained Variance, Weights\n")
results = []

# Goes through the parameters
for lr in learning_rates:
    for n_iter in max_iters:
        for avg in average_options:
            for stop in stop_values:
                model = SGDRegressor(learning_rate='constant', eta0=lr, max_iter=n_iter, tol=stop, average=avg)
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                r2 = r2_score(y_test, y_test_pred)
                explained_variance = explained_variance_score(y_test, y_test_pred)
                weights = model.coef_

                results.append({
                    'learning_rate': lr,
                    'max_iter': n_iter,
                    'average': avg,
                    'stop': stop,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'r2_score': r2,
                    'explained_variance': explained_variance,
                    'weights': weights
                })

                # Log 
                log_file.write(f"{lr}, {n_iter}, {avg}, {stop}, {train_mse:.4f}, {test_mse:.4f}, {r2:.4f}, {explained_variance:.4f}, {weights.tolist()}\n")

log_file.close()

# Best Parameters
best_learning_rate = 0.001
best_max_iter = 1000
best_average = False
best_stop_value = 0.001

best_model = SGDRegressor(learning_rate='constant', eta0=best_learning_rate, max_iter=best_max_iter, tol=best_stop_value, average=best_average)
best_model.fit(X_train, y_train)

y_train_pred_best = best_model.predict(X_train)
y_test_pred_best = best_model.predict(X_test)

# Evaluation metrics for best model
best_train_mse = mean_squared_error(y_train, y_train_pred_best)
best_test_mse = mean_squared_error(y_test, y_test_pred_best)
best_r2 = r2_score(y_test, y_test_pred_best)
best_explained_variance = explained_variance_score(y_test, y_test_pred_best)
best_weights = best_model.coef_

# Output for best model
print("\n|=== Best Model Results ===|")
print(f"Model Test MSE: {best_test_mse:.4f}")
print(f"Model R²: {best_r2:.4f}")
print(f"Model Explained Variance: {best_explained_variance:.4f}")
print(f"Model Weights (Coefficients): {best_weights}")
print("|=== Hyperparameters ===|")
print(f"Learning Rate: {best_learning_rate}")
print(f"Max Iterations: {best_max_iter}")
print(f"Average: {best_average}")
print(f"Stop Value: {best_stop_value}")

# Plot Predicted vs Actual values for the best model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred_best, alpha=0.7)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

# Test MSE vs Number of Iterations for Different Learning Rates (based on exploratory results)
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

#MSE vs Learning Rate
plt.figure(figsize=(10, 6))
learning_rate_values = [result['learning_rate'] for result in results]
test_mse_values = [result['test_mse'] for result in results]
iteration_values = [result['max_iter'] for result in results]

plt.scatter(learning_rate_values, test_mse_values, c=iteration_values, cmap='viridis')
plt.colorbar(label='Max Iterations')
plt.xlabel('Learning Rate')
plt.ylabel('Test MSE')
plt.title('Test MSE for Different Learning Rates and Iterations')
plt.show()

# Target Variable (Temperature, T) vs CO(GT)
plt.figure(figsize=(10, 6))
plt.scatter(df['CO(GT)'], y, alpha=0.5)
plt.xlabel("CO(GT)")
plt.ylabel("Temperature (T)")
plt.title("Temperature vs CO(GT)")
plt.show()

# Target Variable (Temperature, T) vs NOx(GT)
plt.figure(figsize=(10, 6))
plt.scatter(df['NOx(GT)'], y, alpha=0.5, color='green')
plt.xlabel("NOx(GT)")
plt.ylabel("Temperature (T)")
plt.title("Temperature vs NOx(GT)")
plt.show()


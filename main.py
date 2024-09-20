import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'


# loading dataset + cleaning 
housing = fetch_california_housing()

df = pd.DataFrame(housing.data,columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# seperating features (x) and the target (y)
y = df['MedHouseVal']
x = df.drop('MedHouseVal', axis=1)

# random_state: ensures that everytime I run the code it the data points are the same
# test_size: 0.2 = 20% of all the datapoints will be allocated for testing and the rest 80% will be allocated for training
# x: features/ the input which which will be used to make the prediction
# y: target/ouput which will be the prediction we made base don x
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform the training data
x_train_scaled = scaler.fit_transform(x_train)

# Transform the test data using the same scaler (no fitting on test data)
x_test_scaled = scaler.transform(x_test)

# print(x_train_scaled)
# print(x_test_scaled)

# Initialize the Sequential model
model = Sequential()

# Add the input layer and a hidden layer with 64 neurons and ReLU activation
model.add(Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)))

# print(x_train_scaled.shape[1])

# Add another hidden layer with 32 neurons and ReLU activation
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))

# Add the output layer (for regression, we use a single neuron without activation)
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

try:
    # Train the model
    hello = model.fit(x_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(x_test_scaled, y_test))

    # Evaluate the model
    test_loss, test_mae = model.evaluate(x_test_scaled, y_test)
    print(f"Test Mean Absolute Error: {test_mae}")
except Exception as e:
    print(f"An error occurred: {e}")


print("x_train_scaled shape:", x_train_scaled.shape)  # Should be (number of samples, number of features)
print("y_train shape:", y_train.shape)                # Should be (number of samples,)
print("x_test_scaled shape:", x_test_scaled.shape)    # Should be (number of samples, number of features)
print("y_test shape:", y_test.shape)                  # Should be (number of samples,)

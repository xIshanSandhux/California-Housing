import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


# Convert the data to PyTorch tensors
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the neural network model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(x_train_tensor.shape[1], 64)  # Input layer
        self.fc2 = nn.Linear(64, 64)                       # Hidden layer
        self.fc3 = nn.Linear(64, 32)                       # Hidden layer
        self.fc4 = nn.Linear(32, 1)                        # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize the model, loss function, and optimizer
model = NeuralNet()
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)


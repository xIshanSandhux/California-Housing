import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


print(x)
print(y)

# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# Load the dataset
dataset = pd.read_csv('titanic.csv')
# X = dataset.drop(dataset.columns[1], axis=1).values
# y = dataset.iloc[:, 1].values

# Identify the categorical data
columns_to_encode = ['Sex', 'Embarked', 'Pclass']
# columns_to_encode = [1, 3, 10]


# Implement an instance of the ColumnTransformer class
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), columns_to_encode)], remainder='passthrough')

# Apply the fit_transform method on the instance of ColumnTransformer


# Convert the output into a NumPy array
# X = np.array(ct.fit_transform(X))
X = np.array(ct.fit_transform(dataset))

# Use LabelEncoder to encode binary categorical data

le = LabelEncoder()  # passing nothing because we have only one single vector
y = le.fit_transform(dataset['Survived'])
# y = le.fit_transform(y)

# Print the updated matrix of features and the dependent variable vector
print(X)
print(y)

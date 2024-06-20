# Importing the necessary libraries
import pandas as pd

# Loading the Iris dataset
dataset = pd.read_csv('iris.csv')

# Creating the matrix of features (X) and the dependent variable vector (y)
X = dataset.iloc[:, :-1].values # Get all columns except last one
print (X)

# Printing the matrix of features and the dependent variable vector
y = dataset.iloc[:, -1].values
print(y)
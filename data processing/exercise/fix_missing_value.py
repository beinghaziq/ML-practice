# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
# Load the dataset
dataset = pd.read_csv("pima-indians-diabetes.csv")
# Identify missing data (assumes that missing data is represented as NaN)
x = dataset.iloc[:, :-1].values

# Print the number of missing entries in each column

# Configure an instance of the SimpleImputer class

# Fit the imputer on the DataFrame

# Apply the transform to the DataFrame

#Print your updated matrix of features

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, :-1])
x[:, :-1] = imputer.transform(x[:, :-1])
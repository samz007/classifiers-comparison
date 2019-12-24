import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Importing the dataset
dataset = pd.read_csv('../../dataset/winequality-red.csv', encoding="ISO-8859-1")
print(dataset.info())

# finding misssing values in given data-sets
missing = dataset.isnull().sum()
print("\n Missing valuses are {}".format(missing))

# As there are no missing values imputing step is not required, lets print final data-type count present in dataset
print("\n Dtype counts in given dataset: \n{}".format(dataset.dtypes.value_counts()))

# splitting dataset into features and classifier-output
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# splitting data-set into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("\n Test set and training set are separated")

# feature-scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

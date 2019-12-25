import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# printing analysis to txt file
orig_stdout = sys.stdout
f = open("svc_analysis.txt", 'w')
sys.stdout = f

# Importing the dataset for red_wine and white wine
dataset_red = pd.read_csv('..\winequality-red.csv', encoding="ISO-8859-1")
dataset_red.insert(0, 'color', 0)
dataset_white = pd.read_csv('..\winequality-white.csv', encoding="ISO-8859-1")
dataset_white.insert(0, 'color', 0)

# combning two datasets
dataset = pd.concat([dataset_red, dataset_white], axis=0)
print("==========================================================================")

# splitting dataset into features and classifier-output
X = dataset.iloc[:, [0, 2, 5, 8, 11]].values
y = dataset.iloc[:, -1].values

# splitting data-set into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature-scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# fitting classifier for the training set
classifier = SVC(C=10, gamma=0.8, kernel='rbf', random_state=0)

print("==========================================================================")
print("Analysis for classifier with default values : {}".format(np.str(classifier).split('(')[0]))
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
f1_score_for_classifier = f1_score(y_test, y_pred, average='micro')
print("F1 score is {}".format(f1_score_for_classifier))
# applying k-fold cross validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=3)
print("accuracies are {}".format(accuracies))
print(
    "accuracy values on k-fold cross validation have mean as   {} and std as  {}".format(
        round(accuracies.mean(), 6),
        round(accuracies.std(), 2)))

# performing grid-search
from sklearn.model_selection import GridSearchCV

parameters = [{'C': [1, 10, 100, 100], 'kernel': ['linear']},
              {'C': [1, 5, 10, 100, 200, 500], 'kernel': ['rbf'],
               'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8]}]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=3, n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_params = grid_search.best_params_

print("Best accuracy obtained is {}".format(best_accuracy))
print(best_accuracy
      )
print("With params  {}".format(best_params))
sys.stdout = orig_stdout
f.close()

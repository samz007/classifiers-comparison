import sys

import numpy as np
import pandas as pd
from PyXGBoost import PyXGBoostClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# printing analysis to txt file
orig_stdout = sys.stdout
f = open('results\data_analysis_red_wine.txt', 'w')
sys.stdout = f

# Importing the dataset for red_wine
dataset = pd.read_csv('../../dataset/winequality-red.csv', encoding="ISO-8859-1")
print("==========================================================================")
print(np.str(dataset.info()))

# finding misssing values in given data-sets
missing = dataset.isnull().sum()
print("missing values in dataset are \n {}".format(missing))
print("==========================================================================")

# As there are no missing values imputing step is not required, lets print final data-type count present in dataset
print("Dtype counts in given dataset: \n{}".format(dataset.dtypes.value_counts()))

# splitting dataset into features and classifier-output
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# splitting data-set into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("Test set and training set are separated")

# feature-scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# fitting classifier for the training set
# selecting classifiers from sklearn module
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

xgb_classifier = PyXGBoostClassifier()
rf_cassifier = RandomForestClassifier(n_estimators=100, criterion='entropy')
nb_classifier = GaussianNB()
gb_classifier = GradientBoostingClassifier()
svc_classifier = SVC(kernel='rbf', random_state=0, gamma='auto')
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)

classifiers_with_default_values = [xgb_classifier, rf_cassifier, nb_classifier, gb_classifier, svc_classifier,
                                   dt_classifier]

for classifier in classifiers_with_default_values:
    print("==========================================================================")
    print("Analysis for classifier : {}".format(np.str(classifier).split('(')[0]))
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

sys.stdout = orig_stdout
f.close()

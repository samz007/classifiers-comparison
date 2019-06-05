import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

np.set_printoptions(threshold=sys.maxsize)
# printing analysis to txt file
orig_stdout = sys.stdout
f = open("rf_svc_combined_analysis.txt", 'w')
sys.stdout = f

# Importing the dataset for red_wine and white wine
dataset_red = pd.read_csv('..\winequality-red.csv', encoding="ISO-8859-1")
dataset_red.insert(0, 'color', 1)
dataset_white = pd.read_csv('..\winequality-white.csv', encoding="ISO-8859-1")
dataset_white.insert(0, 'color', 0)

# combning two datasets
dataset = pd.concat([dataset_red, dataset_white], axis=0)
print("==========================================================================")

# splitting dataset into features and classifier-output
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# splitting data-set into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature-scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# fitting classifier for the training set
print("==========================================================================")
classifier_rf = RandomForestClassifier(n_estimators=200, criterion='entropy')
print("Analysis for classifier with default values : {}".format(np.str(classifier_rf).split('(')[0]))
classifier_rf.fit(X_train, y_train)
y_pred_rf = classifier_rf.predict(X_train)
cm_rf = confusion_matrix(y_true=y_train, y_pred=y_pred_rf)
f1_score_for_classifier = f1_score(y_train, y_pred_rf, average='micro')
print(cm_rf)
print("F1 score is {}".format(f1_score_for_classifier))
print("==========================================================================")
classifier_svc = SVC(C=10, gamma=0.8, kernel='rbf')
print("Analysis for classifier with default values : {}".format(np.str(classifier_svc).split('(')[0]))
classifier_svc.fit(X_train, y_train)
y_pred_svc = classifier_svc.predict(X_train)
cm_svc = confusion_matrix(y_true=y_train, y_pred=y_pred_svc)
f1_score_for_classifier = f1_score(y_train, y_pred_svc, average='micro')
print(cm_svc)
print("F1 score is {}".format(f1_score_for_classifier))
print("==========================================================================")
# X_combined = np.ma.concatenate([y_pred_svc, y_pred_rf])
X_combined_train = np.column_stack((classifier_svc.predict(X_train),
                                    classifier_rf.predict(X_train)))
X_combined_test = np.column_stack((classifier_svc.predict(X_test),
                                   classifier_rf.predict(X_test)))
rf_classifier_combined = RandomForestClassifier(criterion='entropy')
rf_classifier_combined.fit(X_combined_train, y_train)
y_pred_combnd_test = rf_classifier_combined.predict(X_combined_test)
print(np.shape((y_pred_combnd_test[y_pred_combnd_test != y_test])))

cm_combined = confusion_matrix(y_test, y_pred_combnd_test)
print("confusion matrix for combination is :\n{}".format(cm_combined))
f1_score_for_combined_classifier = f1_score(y_test, y_pred_combnd_test, average='micro')
print("F1 score for combined is {}".format(f1_score_for_combined_classifier))

# applying k-fold cross validation
accuracies_combined = cross_val_score(estimator=rf_classifier_combined, X=X_combined_train, y=y_train, cv=3)
print("accuracies for combined model are {}".format(accuracies_combined))
print(
    "accuracy values on k-fold cross validation have mean as   {} and std as  {}".format(
        round(accuracies_combined.mean(), 6),
        round(accuracies_combined.std(), 2)))

sys.stdout = orig_stdout
f.close()

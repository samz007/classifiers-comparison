import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import seaborn as sns
import sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
np.set_printoptions(threshold=sys.maxsize)
# printing analysis to txt file
orig_stdout = sys.stdout
f = open("feature_selection_analysis.txt", 'w')
sys.stdout = f

# Importing the dataset for red_wine and white wine
dataset_red = pd.read_csv('../../dataset/winequality-red.csv', encoding="ISO-8859-1")
dataset_red.insert(0, 'color', 1)
dataset_white = pd.read_csv('../../dataset/winequality-white.csv', encoding="ISO-8859-1")
dataset_white.insert(0, 'color', 0)

# combning two datasets
dataset = pd.concat([dataset_red, dataset_white], axis=0)
print("==========================================================================")

# splitting dataset into features and classifier-output
X = dataset.iloc[:, [2,5,8, 11]]
y = dataset.iloc[:, -1]
max_features = 1


#get correlations of each features in dataset
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(11,11))
#plot heat map
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# splitting data-set into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=0)


#feature-selection
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=max_features)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#naming the dataframe columns
featureScores.columns = ['Specs','Score']
#print 10 best features
print("Best features obtained with their effective contribution : \n ")
print(featureScores.nlargest(3,'Score'))

# feature-scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
plt.savefig('HeatMap.png')
print("==========================================================================")


sys.stdout = orig_stdout
f.close()

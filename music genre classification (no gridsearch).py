

# DATA CLEANING - https://colab.research.google.com/drive/1DtGtD_qDi6UIFvb0HTd6mllK5VgLA2cK?authuser=1
# MULTICLASSIFICATION - https://colab.research.google.com/drive/14BLwpfnPAqcnhTpOagS9ZjnwPMSnGRxG?authuser=1
# prediction (array) and model evaluation - https://colab.research.google.com/drive/1HgjG3z-lg5nY7f2VkqBzLJBhhlFH6fk7?authuser=1
import time
#Importing libraries for classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import Axes3D

#read dataset
data = pd.read_csv('feature (200).csv',encoding='UTF8')
data = data.iloc[0:, 1:] #drop first row

y = data['label'] # store actual labels in y
X = data.loc[:, data.columns != 'label'] #store data points in X

#Data normalization (scaling)
cols = X.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled, columns = cols)

'''
#### PCA 2 COMPONENTS ####
from sklearn.decomposition import PCA


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

# concatenate with target label
finalDf = pd.concat([principalDf, y], axis = 1)

pca.explained_variance_ratio_
plt.figure(figsize = (16, 9))
sns.scatterplot(x = "principal component 1", y = "principal component 2", data = finalDf, hue = "label", alpha = 0.7,
               s = 100);

plt.title('PCA on Genres', fontsize = 25)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 10);
plt.xlabel("Principal Component 1", fontsize = 15)
plt.ylabel("Principal Component 2", fontsize = 15)
plt.savefig("PCA Scattert.jpg")
'''

#Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# MODEL 1 : K NEAREST NEIGHBORS
start_time = time.time()
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': [1]}
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=100, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
preds = best_model.predict(X_test) #array of predicted genres using KNN 
best_params = grid_search.best_params_
print("Optimal parameters for K nearest neighbors:", best_params)
end_time = time.time()
runtime = end_time - start_time
print("Runtime:", runtime, "seconds")

'''
#Confusion matrix for KNN
cm = confusion_matrix(y_test, preds)
row_sums = cm.sum(axis=1)
normalized_cm = cm / row_sums[:, np.newaxis]
plt.figure(figsize=(9, 9))
sns.heatmap(normalized_cm, annot=True, fmt=".2%", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Normalized Confusion Matrix (K Nearest Neighbors)")
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kulintang", "kundiman", "manila", "pop", "rap", "rock", "rondalla"])
plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kulintang", "kundiman", "manila", "pop", "rap", "rock", "rondalla"])
plt.show()
#Classification report
report = classification_report(y_test,preds)
print(report)
'''

'''
# MODEL 2 : DECISION TREES
start_time = time.time()
tree = DecisionTreeClassifier()
#Grid search using cross validation
param_grid = {'max_depth': [60]}
grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=100, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
preds = best_model.predict(X_test) #array of predicted genres using KNN 
best_params = grid_search.best_params_
print("Optimal parameters for decision trees:", best_params)
end_time = time.time()
runtime = end_time - start_time
print("Runtime:", runtime, "seconds")
#Confusion matrix for decision tree
cm = confusion_matrix(y_test, preds)
row_sums = cm.sum(axis=1)
normalized_cm = cm / row_sums[:, np.newaxis]
plt.figure(figsize=(9, 9))
sns.heatmap(normalized_cm, annot=True, fmt=".2%", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Normalized Confusion Matrix (Decision Tree)")
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kulintang", "kundiman", "manila", "pop", "rap", "rock", "rondalla"])
plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kulintang", "kundiman", "manila", "pop", "rap", "rock", "rondalla"])
plt.show()
#Classification report
report = classification_report(y_test, preds)
print(report)
'''

'''
# MODEL 3 : RANDOM FOREST
start_time = time.time()
rforest = RandomForestClassifier(random_state=0)
#Grid search using cross validation
param_grid = {'max_depth': [20],
              'n_estimators' : [700]
             }
grid_search = GridSearchCV(estimator=rforest, param_grid=param_grid, cv=130, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
preds = best_model.predict(X_test) #array of predicted genres using Random Forest
best_params = grid_search.best_params_
print("Optimal parameters for random forest:", best_params)
end_time = time.time()
runtime = end_time - start_time
print("Runtime:", runtime, "seconds")
#Confusion matrix for random forest
cm = confusion_matrix(y_test, preds)
row_sums = cm.sum(axis=1)
normalized_cm = cm / row_sums[:, np.newaxis]
plt.figure(figsize=(9, 9))
sns.heatmap(normalized_cm, annot=True, fmt=".2%", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Normalized Confusion Matrix (Decision Tree)")
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kulintang", "kundiman", "manila", "pop", "rap", "rock", "rondalla"])
plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kulintang", "kundiman", "manila", "pop", "rap", "rock", "rondalla"])
plt.show()
#Classification report
report = classification_report(y_test, preds)
print(report)
'''

'''
# MODEL 4 : SUPPORT VECTOR MACHINE
start_time = time.time()
svm = SVC()
#Grid search using cross validation
param_grid = {'C': [10],'kernel' : ['rbf']}
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=100 , scoring='accuracy',  error_score='raise')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
preds = best_model.predict(X_test) #array of predicted genres using Random Forest
cv_results = grid_search.cv_results_
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    print("Parameters:", params)
    print("Mean Score:", round(mean_score, 5))
    print("------")
best_params = grid_search.best_params_
print("Optimal parameters for random forest:", best_params)
end_time = time.time()
runtime = end_time - start_time
print("Runtime:", runtime, "seconds")
#Confusion matrix for random forest
cm = confusion_matrix(y_test, preds)
row_sums = cm.sum(axis=1)
normalized_cm = cm / row_sums[:, np.newaxis]
plt.figure(figsize=(9, 9))
sns.heatmap(normalized_cm, annot=True, fmt=".2%", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Normalized Confusion Matrix (svm)")
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kulintang", "kundiman", "manila", "pop", "rap", "rock", "rondalla"])
plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kulintang", "kundiman", "manila", "pop", "rap", "rock", "rondalla"])
plt.show()
#Classification report
report = classification_report(y_test,preds)
print(report)
'''

'''
start_time = time.time()
lg = LogisticRegression(random_state=0, solver='lbfgs', multi_class="multinomial")
# MODEL 5 : LOGISTIC REGRESSION
# Grid search using cross validation
param_grid = {'max_iter': [1000]
             }
grid_search = GridSearchCV(estimator=lg, param_grid=param_grid, cv=100 , scoring='accuracy',  error_score='raise')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
preds = best_model.predict(X_test) #array of predicted genres using Random Forest
cv_results = grid_search.cv_results_
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    print("Parameters:", params)
    print("Mean Score:", round(mean_score, 5))
    print("------")
best_params = grid_search.best_params_
print("Optimal parameters for logistic regression:", best_params)

end_time = time.time()
runtime = end_time - start_time
print("Runtime:", runtime, "seconds")
#Confusion matrix for random forest
cm = confusion_matrix(y_test, preds)
row_sums = cm.sum(axis=1)
normalized_cm = cm / row_sums[:, np.newaxis]
plt.figure(figsize=(9, 9))
sns.heatmap(normalized_cm, annot=True, fmt=".2%", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Normalized Confusion Matrix (Logistic Regression)")
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kulintang", "kundiman", "manila", "pop", "rap", "rock", "rondalla"])
plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kulintang", "kundiman", "manila", "pop", "rap", "rock", "rondalla"])
plt.show()
#Classification report
report = classification_report(y_test,preds)
print(report)
'''


import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt

perm = PermutationImportance(estimator=best_model, random_state=1)
perm.fit(X_test, y_test)

feature_names = X_test.columns.tolist()
feature_importance = perm.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.show()
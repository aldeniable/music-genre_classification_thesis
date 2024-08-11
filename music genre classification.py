

# DATA CLEANING - https://colab.research.google.com/drive/1DtGtD_qDi6UIFvb0HTd6mllK5VgLA2cK?authuser=1
# MULTICLASSIFICATION - https://colab.research.google.com/drive/14BLwpfnPAqcnhTpOagS9ZjnwPMSnGRxG?authuser=1
# prediction (array) and model evaluation - https://colab.research.google.com/drive/1HgjG3z-lg5nY7f2VkqBzLJBhhlFH6fk7?authuser=1

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

#data cleaning
#data = data.drop_duplicates(subset='filename',keep='first')

y = data['label'] # store actual labels in y
X = data.loc[:, data.columns != 'label'] #store data points in X

#Data normalization (scaling)
cols = X.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled, columns = cols)

#Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

'''
# MODEL 1 : K NEAREST NEIGHBORS
knn = KNeighborsClassifier()
#Grid search using cross validation
param_grid = {'n_neighbors': [1,2,3,4,5,10]
             }
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=100, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
preds = best_model.predict(X_test) #array of predicted genres using KNN 

cv_results = grid_search.cv_results_
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    print("Parameters:", params)
    print("Mean Score:", round(mean_score, 5))
    print("------")
best_params = grid_search.best_params_
print("Optimal parameters for K nearest neighbors:", best_params)
grid_results = grid_search.cv_results_
param_values = param_grid['n_neighbors']
param_scores = grid_results['mean_test_score']
#Plotting grid search
plt.plot(param_values, param_scores, marker='o')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('Grid Search (KNN: n_neighbors)')
plt.show()

#Plotting learning curve with cv = 100
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, y_train, cv=100, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.figure(figsize=(8, 6))
plt.fill_between(train_sizes, train_mean - train_std,
                 train_mean + train_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_mean - test_std,
                 test_mean + test_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g",
         label="Cross-validation score")
plt.legend(loc="best")
plt.xlabel("Training data")
plt.ylabel("Accuracy")
print(test_mean)
plt.show()

#Confusion matrix for KNN
cm = confusion_matrix(y_test, preds)
row_sums = cm.sum(axis=1)
normalized_cm = cm / row_sums[:, np.newaxis]
plt.figure(figsize=(9, 9))
sns.heatmap(normalized_cm, annot=True, fmt=".2%", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Normalized Confusion Matrix (K Nearest Neighbors)")
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kundiman", "kulintang", "rondalla", "pop", "manila", "rap", "rock"])
plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kundiman", "kulintang", "rondalla", "pop", "manila", "rap", "rock"])
plt.show()

#Classification report
report = classification_report(y_test,preds)
print(report)
'''

'''
# MODEL 2 : DECISION TREES
tree = DecisionTreeClassifier()
#Grid search using cross validation
param_grid = {'max_depth': [1,5,10,15,20,30,40,50,60,100]
             }
grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=100, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
preds = best_model.predict(X_test) #array of predicted genres using KNN 

cv_results = grid_search.cv_results_
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    print("Parameters:", params)
    print("Mean Score:", round(mean_score, 5))
    print("------")
best_params = grid_search.best_params_
print("Optimal parameters for decision trees:", best_params)
grid_results = grid_search.cv_results_
param_values = param_grid['max_depth']
param_scores = grid_results['mean_test_score']
#Plotting grid search
plt.plot(param_values, param_scores, marker='o')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Grid Search (Decision Tree: max_depth)')
plt.show()

#Plotting learning curve with cv = 100
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, y_train, cv=100, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.figure(figsize=(8, 6))
plt.fill_between(train_sizes, train_mean - train_std,
                 train_mean + train_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_mean - test_std,
                 test_mean + test_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g",
         label="Cross-validation score")
plt.legend(loc="best")
plt.title("Learning Curve (Decision Tree)")
plt.xlabel("Training data")
plt.ylabel("Accuracy")
print(test_mean)
plt.show()

#Confusion matrix for decision tree
cm = confusion_matrix(y_test, preds)
row_sums = cm.sum(axis=1)
normalized_cm = cm / row_sums[:, np.newaxis]
plt.figure(figsize=(9, 9))
sns.heatmap(normalized_cm, annot=True, fmt=".2%", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Normalized Confusion Matrix (Decision Tree)")
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kundiman", "kulintang", "rondalla", "pop", "manila", "rap", "rock"])
plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kundiman", "kulintang", "rondalla", "pop", "manila", "rap", "rock"])
plt.show()

#Classification report
report = classification_report(y_test, preds)
print(report)
'''

'''
# MODEL 3 : RANDOM FOREST
rforest = RandomForestClassifier(random_state=0)
#Grid search using cross validation
param_grid = {'max_depth': [20],
              'n_estimators' : [700]
             }
grid_search = GridSearchCV(estimator=rforest, param_grid=param_grid, cv=130, scoring='accuracy')
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
grid_results = grid_search.cv_results_
param_values_depth = param_grid['max_depth']
param_values_estimators = param_grid['n_estimators']
param_scores = grid_results['mean_test_score']

# Create meshgrid for heatmap
X, Y = np.meshgrid(param_values_depth, param_values_estimators)

# Reshape scores to match the meshgrid dimensions
Z = np.array(param_scores).reshape(X.shape)

# Create heatmap
plt.figure(figsize=(10, 6))
plt.imshow(Z, cmap='viridis', origin='lower')

# Set tick labels
plt.xticks(np.arange(len(param_values_depth)), param_values_depth)
plt.yticks(np.arange(len(param_values_estimators)), param_values_estimators)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Accuracy')

# Set labels and title
plt.xlabel('max_depth')
plt.ylabel('n_estimators')
plt.title('Grid Search')

# Show the plot
plt.show()
#Confusion matrix for random forest
cm = confusion_matrix(y_test, preds)
row_sums = cm.sum(axis=1)
normalized_cm = cm / row_sums[:, np.newaxis]
plt.figure(figsize=(9, 9))
sns.heatmap(normalized_cm, annot=True, fmt=".2%", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Normalized Confusion Matrix (Decision Tree)")
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kundiman", "kulintang", "rondalla", "pop", "manila", "rap", "rock"])
plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kundiman", "kulintang", "rondalla", "pop", "manila", "rap", "rock"])
plt.show()
#Classification report
report = classification_report(y_test, preds)
print(report)
'''

'''
# MODEL 4 : SUPPORT VECTOR MACHINE
svm = SVC()
#Grid search using cross validation
param_grid = {'C': [1,2,3,4,5,6,7,8,9,10,50],
              'kernel' : ['poly','rbf','linear','sigmoid']
             }
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=10 , scoring='accuracy',  error_score='raise')
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
grid_results = grid_search.cv_results_
param_values_C = param_grid['C']
param_values_kernel = param_grid['kernel']
param_scores = grid_results['mean_test_score']

# Create meshgrid for heatmap
X, Y = np.meshgrid(param_values_C, param_values_kernel)

# Reshape scores to match the meshgrid dimensions
Z = np.array(param_scores).reshape(X.shape)

# Create heatmap
plt.figure(figsize=(10, 6))
plt.imshow(Z, cmap='viridis', origin='lower')

# Set tick labels
plt.xticks(np.arange(len(param_values_C)), param_values_C)
plt.yticks(np.arange(len(param_values_kernel)), param_values_kernel)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Accuracy')

# Set labels and title
plt.xlabel('C')
plt.ylabel('kernel')
plt.title('Grid Search')

# Show the plot
plt.show()
#Confusion matrix for random forest
cm = confusion_matrix(y_test, preds)
row_sums = cm.sum(axis=1)
normalized_cm = cm / row_sums[:, np.newaxis]
plt.figure(figsize=(9, 9))
sns.heatmap(normalized_cm, annot=True, fmt=".2%", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Normalized Confusion Matrix (Decision Tree)")
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kundiman", "kulintang", "rondalla", "pop", "manila", "rap", "rock"])
plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["kundiman", "kulintang", "rondalla", "pop", "manila", "rap", "rock"])
plt.show()
#Classification report
report = classification_report(y_test,preds)
print(report)
'''


lg = LogisticRegression(random_state=0, solver='lbfgs', multi_class="multinomial")
# MODEL 5 : LOGISTIC REGRESSION
# Grid search using cross validation
param_grid = {'max_iter': [1000,1500,2000]
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
grid_results = grid_search.cv_results_
param_values = param_grid['max_depth']
param_scores = grid_results['mean_test_score']
#Plotting grid search
plt.plot(param_values, param_scores, marker='o')
plt.xlabel('max_iter')
plt.ylabel('Accuracy')
plt.title('Grid Search (Decision Tree: max_iter)')
plt.show()

#Plotting learning curve with cv = 100
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, y_train, cv=100, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.figure(figsize=(8, 6))
plt.fill_between(train_sizes, train_mean - train_std,
                 train_mean + train_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_mean - test_std,
                 test_mean + test_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g",
         label="Cross-validation score")
plt.legend(loc="best")
plt.title("Learning Curve (Logistic Regression)")
plt.xlabel("Training data")
plt.ylabel("Accuracy")
print(test_mean)
plt.show()

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
from IPython.display import display
from IPython.display import HTML
perm = PermutationImportance(estimator=knn, random_state=1)
perm.fit(X_test, y_test)
print(eli5.format_as_text(eli5.show_weights(estimator=perm, feature_names=X_test.columns.tolist())))
'''
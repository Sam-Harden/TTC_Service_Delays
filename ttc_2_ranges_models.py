#!/usr/bin/env python
# coding: utf-8


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

import pickle

import warnings
warnings.simplefilter(action='ignore', category=Warning)


# Load the prepared data from the file
fl = '3. Prepared Data/ttc_2_ranges_prepared_data.xlsx'
data=pd.read_excel(fl)

print(f"\nThe file '{fl}' loaded.")
print(f'\nPrepared Data: {data.shape}')

print('Data:')
print(data.head())
print(f'\nDelay ranges:')
print(data['Delay_Time_Range'].value_counts())

# Set 1 to 60 minetes and more delays in 'Delay_Time_Range'
data['Delay_Time_Range']=data['Delay_Time_Range'].apply(lambda x: 1 if x=='[60, 516)' else 0)
print(data['Delay_Time_Range'].value_counts())


#----------------------------  Helper functions --------------------------------------

########################################################################################

def culculate_cross_validate_scores(models, names):
    """ Estimates the accuracy of a model by computing the multiple metrics and its average using cross_validate function. """
        
    # A list of metrics 
    scoring = ['recall', 'f1']
    # A list of relevant metrics value names
    #scoring_results = ['test_recall', 'test_f1', 'fit_time']
    scoring_results = ['test_recall', 'test_f1']
     
    for model, name in zip(models, names):
        print(name,':')
        # Compute the multiple metrics and its averages using cross_validate function
        results = cross_validate(model, X, y, scoring=scoring, cv=10, return_train_score=False)
        # Print metrics and its averages
        for metric_name in results.keys():
            if metric_name in scoring_results:
                print(metric_name)
                print(results[metric_name])
                # Print metrics' averages
                average_score = np.average(results[metric_name])
                print(f'Average  {metric_name}: {average_score}\n')  
                
                
def get_confusion_matrix_classification_report(model, X_train, y_train, X_test, y_test):
    """ Calculates the confusion matrix and classification report of the model. """
    # Fit the model
    model.fit(X_train, y_train)
    # Make predictions
    y_predict = model.predict(X_test)
    
    # Generate the confusion matrix and classification report
    print(confusion_matrix(y_test,y_predict)) 
    print(classification_report(y_test,y_predict))


# ------------------  Evaluation of classification models  --------------------------- 
print('\nEvaluation of classification models\n')
########################################################################################
# Create features and target from the data set
X = data.drop(['Delay_Time_Range'],axis=1)
y = data['Delay_Time_Range']

#####  --------------- Logistic Regression model ----------------------------
lr=LogisticRegression(solver='liblinear', multi_class='ovr', class_weight = 'balanced')


#####  --------------- Random Forest Classifier model ------------------------
rf=RandomForestClassifier(n_estimators=10, class_weight = 'balanced')


#####  --------------- Linear Support Vector Classifier model ------------------------
lsvc = LinearSVC(class_weight = 'balanced')

models = [lr, rf, lsvc]
names = ['LogisticRegression', 'RandomForestClassifier', 'LinearSVC']
    
culculate_cross_validate_scores(models, names)


# --------------------  Evaluation of parameters of LinearSVC model  ------------------ 
print('\nEvaluation of parameters of LinearSVC model\n')
########################################################################################

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)

# Define parameters
param_grid = {'loss': ['hinge', 'squared_hinge'], 'C' : [1.0, 0.1, 0.01], 'multi_class' : ['ovr', 'crammer_singer']}
scoring = 'recall'

#####  --------------- LinearSVC model ----------------------------
lsvc = LinearSVC(class_weight = 'balanced', max_iter=10000)

# Create Stratified K-Folds cross-validation object
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
# Search the best parameters over specified parameter values using GridSearchCV estimator
grid_search = GridSearchCV(lsvc, param_grid, cv=kfold, scoring=scoring)

# Fit the grid
grid_search.fit(X_train, y_train)
# Make predictions
grid_predictions = grid_search.predict(X_test)

best_grid_parameters = grid_search.best_params_
best_grid_score = grid_search.best_score_
best_grid_model = grid_search.best_estimator_

print(f'Best parameters: {best_grid_parameters}')
print(f"Best '{scoring}' score: {best_grid_score} using {best_grid_parameters}")
print(f'\nResult model with parameters: \n {best_grid_model} \n')

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)
get_confusion_matrix_classification_report(best_grid_model, X_train, y_train, X_test, y_test)


# Save the model
model = '4. Insights/Models/ttc_2_delays_model.pickle'
with open(model, 'wb') as f:
    pickle.dump(best_grid_model, f)

print(f"The file '{model}' is saved.")
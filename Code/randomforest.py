import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import time

def randomforest(X, y_a, y_t, X_val, y_a_val, y_t_val,divisor):
    X = X.reshape(X.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    
    start = time.time()
    
    ''' HelpDesk '''
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # X_val = scaler.transform(X_val)
    
    # print("Classifier")
    # # For the first output branch (multiclass classification)
    # rf_classifier = RandomForestClassifier(random_state=42,n_estimators=19, max_depth=14)
    # rf_classifier.fit(X, y_a)
    # y1_train_pred_rf = rf_classifier.predict(X_val)
    # accuracy = accuracy_score(y_a_val, y1_train_pred_rf)
    
    # print("regressor")
    # # For the second output branch (regression)
    # rf_regressor = RandomForestRegressor(random_state=42,n_estimators=100, max_depth=9)
    # rf_regressor.fit(X, y_t)
    # y2_train_pred_rf = rf_regressor.predict(X_val)
    # mae = (mean_absolute_error(y_t_val, y2_train_pred_rf)) * divisor / 86400
    
    
    ''' BPI_12_W'''
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_val = scaler.transform(X_val)
    
    param_grid_classifier = {
        'n_estimators': [3,5,10,15,20,30,50],
        'max_depth': [None, 3,4,5],
    }
    
    param_grid_regressor = {
        'n_estimators': [10,20,50,100,150],
        'max_depth': [None, 2,3,5,7,10],
    }
    
    print("Classifier")
    # For the first output branch (multiclass classification)
    rf_classifier = RandomForestClassifier(random_state=42)
    grid_search_classifier = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_classifier, 
                                          cv=2, n_jobs=-1, scoring='accuracy')
    grid_search_classifier.fit(X, y_a)
    best_classifier = grid_search_classifier.best_estimator_
    y1_train_pred_rf = best_classifier.predict(X_val)
    print("Best parameters:",grid_search_classifier.best_params_)
    accuracy = accuracy_score(y_a_val, y1_train_pred_rf)

    # For the second output branch (regression)
    rf_regressor = RandomForestRegressor(random_state=42)
    grid_search_regressor = GridSearchCV(estimator=rf_regressor, param_grid=param_grid_regressor, 
                                         cv=2, n_jobs=-1, scoring='neg_mean_absolute_error')
    grid_search_regressor.fit(X, y_t)
    best_regressor = grid_search_regressor.best_estimator_
    y2_train_pred_rf = best_regressor.predict(X_val)
    print("Best parameters:",grid_search_regressor.best_params_)
    mae = (mean_absolute_error(y_t_val, y2_train_pred_rf)) * divisor / 86400
    
    end = time.time()
    
    return accuracy,mae,end-start
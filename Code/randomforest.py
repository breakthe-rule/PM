import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import time

def randomforest(X, y_a, y_t, X_val, y_a_val, y_t_val,divisor,dataset):
    X = X.reshape(X.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_val = scaler.transform(X_val)
    
    start = time.time()
    
    ''' HelpDesk '''
    if dataset=="helpdesk":
        print("Classifier")
        # For the first output branch (multiclass classification)
        rf_classifier = RandomForestClassifier(random_state=42,n_estimators=19, max_depth=14)
        rf_classifier.fit(X, y_a)
        y1_train_pred_rf = rf_classifier.predict(X_val)
        accuracy = accuracy_score(y_a_val, y1_train_pred_rf)
        
        print("regressor")
        # For the second output branch (regression)
        rf_regressor = RandomForestRegressor(random_state=42,n_estimators=100, max_depth=9)
        rf_regressor.fit(X, y_t)
        y2_train_pred_rf = rf_regressor.predict(X_val)
        mae = (mean_absolute_error(y_t_val, y2_train_pred_rf)) * divisor / 86400
    
    
    ''' BPI_12_W'''
    if dataset=="bpi12":
        print("Classifier")
        # For the first output branch (multiclass classification)
        rf_classifier = RandomForestClassifier(random_state=42,n_estimators=170, max_depth=50)
        rf_classifier.fit(X, y_a)
        y1_train_pred_rf = rf_classifier.predict(X_val)
        accuracy = accuracy_score(y_a_val, y1_train_pred_rf)

        print("regressor")
        # For the second output branch (regression)
        rf_regressor = RandomForestRegressor(random_state=42,n_estimators=350, max_depth=9)
        rf_regressor.fit(X, y_t)
        y2_train_pred_rf = rf_regressor.predict(X_val)
        mae = (mean_absolute_error(y_t_val, y2_train_pred_rf)) * divisor / 86400
    
    ''' BPI_20'''
    if dataset == "bpi20":
        print("Classifier")
        # For the first output branch (multiclass classification)
        rf_classifier = RandomForestClassifier(random_state=42,n_estimators=190, max_depth=40)
        rf_classifier.fit(X, y_a)
        y1_train_pred_rf = rf_classifier.predict(X_val)
        accuracy = accuracy_score(y_a_val, y1_train_pred_rf)

        print("regressor")
        # For the second output branch (regression)
        rf_regressor = RandomForestRegressor(random_state=42,n_estimators=100, max_depth=8)
        rf_regressor.fit(X, y_t)
        y2_train_pred_rf = rf_regressor.predict(X_val)
        mae = (mean_absolute_error(y_t_val, y2_train_pred_rf)) * divisor / 86400
    
    end = time.time()
    
    return accuracy,mae,end-start
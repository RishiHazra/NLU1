# -*- coding: utf-8 -*-
"""
bagging with XGBoost on metadata (Liar Dataset)
@author: Rishi

"""
import pandas as pd
import xgboost as xgb
import numpy as np
import bisect
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier


train_df = pd.read_csv("train_subject.csv", index_col=None)
test_df = pd.read_csv("test_subject.csv", index_col=None)
valid_df= pd.read_csv("valid_subject.csv", index_col=None)

with open('train.tsv',encoding='utf8') as tsvfile:
     train_data = pd.read_csv(tsvfile, delimiter='\t',header=None)
     tsvfile.close()
     
with open('test.tsv',encoding='utf8') as tsvfile:
    test_data=pd.read_csv(tsvfile, delimiter='\t', header=None)
    tsvfile.close()


features=['ID','Label','statement','subject','speaker','job-title','state info','affliation','barely true','false','half true','mostly true','pants on fire','context']
        
Y_test=test_data[1]     
Y_train=train_data[1]

# removing the 'context' feature 
X_train=train_data.iloc[:,4:13]
X_test= test_data.iloc[:,4:13]


# one_hot_encoding the categorical variables
categorical_columns= [4,5,6,7]

# re-index the new data to the columns of the training data
# filling the missing values with 0
'''
dummy_train = pd.get_dummies(train_data.iloc[:,[4,5,6,7]])
dummy_new = pd.get_dummies(test_data.iloc[:,[4,5,6,7]])
dummy_new.reindex(columns = dummy_train.columns, fill_value=0)
le = LabelEncoder()
X_train=np.concatenate((dummy_train,X_train),axis=1)  
X1_train=np.concatenate((train_df,X_train),axis=1)
X_test=np.concatenate((dummy_new,X_test),axis=1)
X1_test=np.concatenate((test_df,X_test),axis=1)

'''
le = LabelEncoder()

Y_train=le.fit_transform(Y_train)
Y_test=le.transform(Y_test)

dummy_train=np.zeros((10240,1))
for index,cat in enumerate(categorical_columns):
    X_train[cat] = le.fit_transform(X_train[cat].fillna('0'))
    X_test[cat] = test_data.iloc[:,cat].map(lambda s: 'other' if s not in le.classes_ else s)
    le_classes = np.array(le.classes_).tolist()
    bisect.insort_left(le_classes, 'other')
    le.classes_ = le_classes
    X_test[cat]= le.transform(X_test[cat])
    

X_train=np.concatenate((train_df,X_train),axis=1)
X_test=np.concatenate((test_df,X_test),axis=1) 

# remove rows with Nan value
mask = ~np.any(np.isnan(X_train), axis=1)
X1_train1 = X_train[mask]
Y1_train1 = Y_train[mask]

dtrain = xgb.DMatrix(X1_train1, label=Y1_train1)
dtest = xgb.DMatrix(X_test)
print(train_df.shape)
print(test_df.shape)

watchlist = [(dtrain, 'train')]

for seed in [1234]:
    param = {'max_depth':3, 
             'eta':0.02, 
             'silent':1, 
             'num_class':6,
             'objective':'multi:softmax',
             'eval_metric': "merror",
             'colsample_bytree': 0.7,
             'booster': "gbtree",
             'seed': seed
             }
    
    num_round = 400
    plst = param.items()
    # bst is the best model for XGBoost
    bst = xgb.train( plst, dtrain, num_round, watchlist )
    
ypred = bst.predict(dtest)

accuracy=0
for i in range(ypred.shape[0]):
    accuracy += int(ypred[i] == Y_test[i]) 
accuracy/ypred.shape[0]




# ============================ Random Forest ==================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def main():
    
    # Create the parameter grid based on the results of random search 
    param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3], 
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
    }
    # Create a based model
    rc = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rc, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

    # remove rows with Nan values from train data
    mask = ~np.any(np.isnan(X_train), axis=1)
    X_train1 = X_train[mask]
    Y_train1 = Y_train[mask]
    
    
    grid_search.fit(X_train1,Y_train1)
    grid_search.best_params_
    # grid search is the best model for Random Forest
    return grid_search

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy=0
    for i in range(predictions.shape[0]):
        accuracy += int(predictions[i] == test_labels[i]) 
    accuracy=accuracy/predictions.shape[0]
    return accuracy

if __name__ == '__main__':
    model = main()
    best_grid = model.best_estimator_    
    evaluate(best_grid,X_test, Y_test)
    
# using best model
rc = RandomForestClassifier(n_estimators=1000, max_depth=80, max_features=3, min_samples_leaf= 3, min_samples_split= 10, bootstrap=True)

rc.fit(X1_train1,Y1_train1)
evaluate(rc, X_test, Y_test)
    

# best parameters for grid search model
    
# {'bootstrap': True,
# 'max_depth': 80,
# 'max_features': 3,
# 'min_samples_leaf': 3,
# 'min_samples_split': 10,
# 'n_estimators': 1000}    
    


# ======================== Combination of the two models ======================#

for alpha in range(0,101):
    combined_model = alpha*bst.predict(dtest) + (100-alpha)*rc.predict(X_test)
    combined_model=np.rint(combined_model/100)
    accuracy=0
    for i in range(ypred.shape[0]):
        accuracy += int(combined_model[i] == Y_test[i]) 
    accuracy=accuracy/ypred.shape[0]
    print(accuracy)

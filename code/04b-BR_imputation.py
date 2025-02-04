import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, recall_score, f1_score, precision_score, confusion_matrix

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge

import os
from datetime import date
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

sns.set(style='whitegrid', rc={"grid.linewidth": 0.5})
font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 22}
plt.rc('font', **font)
plt.rcParams['figure.autolayout']= True # salva los gráficos sin cortat tiítulos de ejes

# Dir creation
# Dir for saving results
# Today
now = str(date.today())

# Parent Directory path
parent_dir = "/home/marcelo/GitRepos/Tesis/"
results_dir = "/home/marcelo/GitRepos/Tesis/pred_results/"

# Path
path = os.path.join(results_dir, now)
try:
    os.makedirs(path)
    print('Directory created!')
except FileExistsError:
    print('Directory already exsist!')
    pass

path_hparams = os.path.join(path, 'BR_hparams')
# path_results = os.path.join(path, 'graphs2')
try:
    os.makedirs(path_hparams)
    print('Directory created!')
except FileExistsError:
    print('Directory already exsist!')
    pass

# Data path
data_path ='./Tesis/data/'
os.getcwd()
train = pd.read_csv("/home/marcelo/GitRepos/Tesis/data/train.csv")
train.isna().sum()[train.isna().sum()>0]

train.columns
# Bayesian Ridge Imputation
imputed_data = train.copy()
diag_dummies = pd.get_dummies(imputed_data['diagnosis'])
imputed_data = pd.concat([imputed_data, diag_dummies], axis=1)
imputed_data = imputed_data.drop(['site', 'id', 'diagnosis', 'strata'], axis=1)
imputer = IterativeImputer(estimator=BayesianRidge(n_iter=1000))
imputer = imputer.fit(imputed_data)
imputed_data = pd.DataFrame(imputer.transform(imputed_data), columns = imputed_data.columns, index=imputed_data.index )
imputed_data.isna().sum()

### Seleccionamos las columnas finales

final_cols = ['site', 'id', 'diagnosis', 'year_birth', 'sex', 'years_education',  'ifs_total_score', 'mini_sea_total', 'npi_total', 'npi_total_caregiver', 'mmse_vs', 'moca_vs','ace_vs', 'functionality', 'marital_status', 'n_children', 'household_members', 'household_income', 'Job_status', 'strata']

final_cols2 = ['diagnosis', 'ifs_total_score', 'mini_sea_total', 'npi_total', 'npi_total_caregiver', 'mmse_vs', 'moca_vs','ace_vs', 'functionality' ]

imputed_data = pd.concat([train[['site', 'id', 'diagnosis', 'strata']], imputed_data], axis=1)
imputed_data = imputed_data[final_cols]

#### Librerías 
os.chdir("/home/marcelo/GitRepos/Tesis/code")
import py_funs
import stats_funs
import ml_hparams_clf as bhs # Bayes search
import ml_bootstrap_clf as bc # boostraping classifiers
from skopt.space import Categorical, Integer, Real 

### Split into AD and FTD
imputed_data.drop(['site','id','strata'], axis=1, inplace=True)
imputed_data = imputed_data.query("diagnosis != 'CN'")
imputed_data['diagnosis'] = imputed_data['diagnosis'].replace({'AD':0, 'FTD':1})
imputed_data.shape, train.shape

## Random Forest Hparams

rf_grid= {"n_estimators": Integer(low=25, high=500),
    "criterion": Categorical(['gini', 'entropy']),
    "max_depth": Integer(low=1, high=10),
    "min_samples_split": Real(low=0.01, high=0.99),
    "min_samples_leaf": Real(low=0.01, high=0.5),
    "max_features":Integer(low=1, high=10)}

rf_param = { "class_weight":"balanced", "verbose":0, "n_jobs":-1}

best, raw = bhs.hparams_search(imputed_data, 'diagnosis', RandomForestClassifier(), rf_grid, rf_param, scaler='MM', test_size= .2, cv=StratifiedKFold(5, shuffle=True), n_iter=100)
print('RF: All Done!')

# Save RF hparams
raw.to_csv(path_hparams + "/BR_RF_hparams.csv")

## Support Vector Machies - Poly params
svc_poly ={'C':Real(low=0.001, high=10),
          'gamma': Real(low=0.001, high=10),
          'degree': Integer(low=1, high=3),
          'coef0': Integer(low=0, high=100)}
svc_poly_param = { "kernel":"poly","class_weight":"balanced", "verbose":0}#, "cache_size":500}

best, raw = bhs.hparams_search(imputed_data, 'diagnosis', SVC(), svc_poly, svc_poly_param, scaler='MM', test_size= .2, cv=StratifiedKFold(5, shuffle=True), n_iter=100)
print('SVC: All Done!')

# Save poly params 
raw.to_csv(path_hparams + "/BR_SVC_Poly_hparams.csv")

## XGBoost - params
xgb_grid = {
    'objective': Categorical(['binary:logistic']),
        'eval_metric': Categorical(['logloss']),
        'n_estimators': Integer(low=100, high=1000),
        'learning_rate': Real(low=0.1, high=0.3),
        'max_depth': Integer(low=2, high=10),
        # 'subsample': Real(0.8, 1.0),
        'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'min_child_weight': [15, 20, 25, 30]
        # 'reg_alpha': Real(0, .8),
        # 'reg_lambda': Real(0, .8),
        # 'gamma': Real(0.001, 10.0)
}

xgb_param = {
    #'enable_categorical':True, # Supported tree methods are `gpu_hist`, `approx`, and `hist`.
    'subsample':1.0, 'gamma': 1.0,
    'n_jobs': -1,
    'verbosity':0,
    'eval_metric':'auc',
    'use_label_encoder':None
}

best, raw = bhs.hparams_search_xgb(imputed_data, 'diagnosis', xgb_grid, xgb_param, scaler='MM', test_size= .2, cv=StratifiedKFold(5, shuffle=True), n_iter=100)
print('Xgboost: All Done!')

# Save XGBoost params 
raw.to_csv(path_hparams + "/BR_xgb_hparams.csv")

###########################################
################## Only test cols
imputed_data2 = imputed_data[final_cols2].copy()

## Random Forest Hparams

rf_grid= {"n_estimators": Integer(low=25, high=500),
    "criterion": Categorical(['gini', 'entropy']),
    "max_depth": Integer(low=1, high=6),
    "min_samples_split": Real(low=0.01, high=0.99),
    "min_samples_leaf": Real(low=0.01, high=0.5),
    "max_features":Integer(low=1, high=6)}

rf_param = { "class_weight":"balanced", "verbose":0, "n_jobs":-1}

#RF
best, raw = bhs.hparams_search(imputed_data2, 'diagnosis', RandomForestClassifier(), rf_grid, rf_param, scaler='MM', test_size= .2, cv=StratifiedKFold(5, shuffle=True), n_iter=100)
print('RF: All Done!')

# Save RF hparams
raw.to_csv(path_hparams + "/BR_RF_hparams_fcols2.csv")

#Poly
best, raw = bhs.hparams_search(imputed_data2, 'diagnosis', SVC(), svc_poly, svc_poly_param, scaler='MM', test_size= .2, cv=StratifiedKFold(5, shuffle=True), n_iter=100)
print('SVC: All Done!')

# Save poly params 
raw.to_csv(path_hparams + "/BR_SVC_Poly_hparams_fcols2.csv")

## XGBoost - params
xgb_grid = {
    'objective': Categorical(['binary:logistic']),
        'eval_metric': Categorical(['logloss']),
        'n_estimators': Integer([100, 1000]),
        'learning_rate': Real(0.1, 0.3),
        'max_depth': Integer(2, 6),
        # 'subsample': Real(0.8, 1.0),
        'colsample_bytree': (0.4, 0.6, 0.8, 1.0),
        'min_child_weight': (10, 15, 20, 25, 30)
        # 'reg_alpha': Real(0, .8),
        # 'reg_lambda': Real(0, .8),
        # 'gamma': Real(0.001, 10.0)
}

xgb_param = {
    #'enable_categorical':True, # Supported tree methods are `gpu_hist`, `approx`, and `hist`.
    'subsample':1.0, 'gamma': 1.0,
    'n_jobs': -1,
    'verbosity':0,
    'eval_metric':'auc',
    'use_label_encoder':None
}

best, raw = bhs.hparams_search_xgb(imputed_data2, 'diagnosis', xgb_grid, xgb_param, scaler='MM', test_size= .2, cv=StratifiedKFold(5, shuffle=True), n_iter=100)
print('Xgboost: All Done!')

# Save XGBoost params 
raw.to_csv(path_hparams + "/BR_xgb_hparams_fcols2.csv")
print("##### Only Test columns  DONE! ##### ##### Only Test columns  DONE! ##### ##### Only Test columns  DONE! #####")
print("################################")
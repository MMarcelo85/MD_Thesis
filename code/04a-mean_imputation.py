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

path_hparams = os.path.join(path, 'mean_hparams')
# path_results = os.path.join(path, 'graphs2')
try:
    os.makedirs(path_hparams)
    print('Directory created!')
except FileExistsError:
    print('Directory already exsist!')
    pass

# # Data path
data_path ='./Tesis/data/'
# # Load data 
# df = pd.read_csv(data_path+'clean_data.csv')
# # Drop participants with no MMSE or MoCA or ACE-III
# df = df.dropna(subset='mmse_vs')
# df.isna().sum()
# df.shape # 1613, 48
# df = df.drop(['aod', 'yod', 'laterality', 'income_sources' ], axis=1)
# df['years_education']=df['years_education'].astype('float')
# df['nationality'].unique()
# nat_dictionary = {'Argentina':1, 'Chilena':2, 'Colombiana': 3, 'Mexicana':4, 'PERUANA':5, 'Española':6, 'Alemana':7}
# # Tiramos un espaqñol y un alemán
# df = df.loc[(df['nationality']!='Española') & (df['nationality']!='Alemana')]
# # Reemplazamos
# df['nationality'] = df['nationality'].replace(nat_dictionary)

# # Idem con Residencia
# df['country_of_residence'].unique()
# res_dictionary = {'Argentina':1, 'Chile':2, 'Colombia': 3, 'Mexico':4, 'Peru':5, 'Panamá':6}
# df['country_of_residence'] = df['country_of_residence'].replace(res_dictionary)


# # Salvamos el df a modelar
# df.to_csv("./Tesis/data/data_to_model.csv", index=False)
# df.info()
os.getcwd()
train = pd.read_csv("./Tesis/data/train.csv")
# test = pd.read_csv("./Tesis/data/test.csv")
train.isna().sum()[train.isna().sum()>0]
# test.isna().sum()[test.isna().sum()>0]

def impute_mean(group):
    return group.fillna(group.mean())


# impute mean by Diagnosis group
# Train
grouped_train = train.groupby('diagnosis')
nan_cols = train.isna().sum()[train.isna().sum()>0].index.tolist()

for i in nan_cols:
    train[i] = train[i].transform(impute_mean)

assert(train.isna().sum().sum() == 0)

#Test
# grouped_test = test.groupby('diagnosis')
# nan_cols = test.isna().sum()[test.isna().sum()>0].index.tolist()

# for i in nan_cols:
#     test[i] = test[i].transform(impute_mean)

# assert(train.isna().sum().sum() == 0)
# assert(test.isna().sum().sum() == 0)

### Tiramos contry of residence y nationality
train.drop(['nationality', 'country_of_residence'], axis=1, inplace=True)
# test.drop(['nationality', 'country_of_residence'], axis=1, inplace=True)


#### Librerías 
os.chdir("/home/marcelo/GitRepos/Tesis/code")
import py_funs
import stats_funs
import ml_hparams_clf as bhs # Bayes search
import ml_bootstrap_clf as bc # boostraping classifiers
from skopt.space import Categorical, Integer, Real 

# ####   Separate df between train and test (25%)
# # Make strata column
# df['strata'] = df.diagnosis + " / " + df.site

# test = py_funs.stratified_sample(df, ['strata'], size=int(df.shape[0]*.20), seed=123, keep_index=True)
# test = test.set_index('index')
# test.index.name = None
# train = df.drop(test.index, axis=0)
# train.shape, test.shape

# df.diagnosis.value_counts()/df.shape[0]
# train.diagnosis.value_counts()/train.shape[0]
# test.diagnosis.value_counts()/test.shape[0]

# #save train and test 
# train.to_csv("../data/train.csv", index=False)
# test.to_csv("../data/test.csv", index=False)

# train = train.drop(['site','id','strata'], axis=1)
# test = test.drop(['site','id','strata'], axis=1)

### Split into AD and FTD
train.drop(['site','id','strata'], axis=1, inplace=True)
data = train.query("diagnosis != 'CN'")
data['diagnosis'] = data['diagnosis'].replace({'AD':0, 'FTD':1})
train.shape, data.shape

## Random Forest Hparams

rf_grid= {"n_estimators": Integer(low=25, high=500),
    "criterion": Categorical(['gini', 'entropy']),
    "max_depth": Integer(low=1, high=14),
    "min_samples_split": Real(low=0.01, high=0.99),
    "min_samples_leaf": Real(low=0.01, high=0.5),
    "max_features":Integer(low=1, high=14)}

rf_param = { "class_weight":"balanced", "verbose":0, "n_jobs":-1}

best, raw = bhs.hparams_search(data, 'diagnosis', RandomForestClassifier(), rf_grid, rf_param, scaler='none', test_size= .2, cv=StratifiedKFold(5, shuffle=True), n_iter=100)
print('All Done!')
# Save RF hparams
raw.to_csv(path_hparams + "/RF_hparams.csv")

## Support Vector Machies - RBF params
svc_rbf ={'C':Real(low=0.001, high=10),
          'gamma': Real(low=0.001, high=10)}
svc_rbf_param = { "kernel":"rbf","class_weight":"balanced", "verbose":0}#, "cache_size":500}

# Cache size its bugged since ever -> https://github.com/scikit-learn/scikit-learn/issues/8012
# Hardcoded on /home/marcelo/anaconda3/envs/sklearn-env/lib/python3.8/site-packages/sklearn/svm/_classes.py
# cache_size=4000

best, raw = bhs.hparams_search(data, 'diagnosis', SVC(), svc_rbf, svc_rbf_param, scaler='MM', test_size= .2, cv=StratifiedKFold(5, shuffle=True), n_iter=100)
print('All Done!')

# Save SVC RBF params 
raw.to_csv(path_hparams + "/SVC_RBF_hparams2.csv")

## Support Vector Machies - Poly params
svc_poly ={'C':Real(low=0.001, high=10),
          'gamma': Real(low=0.001, high=10),
          'degree': Integer(low=1, high=3),
          'coef0': Integer(low=0, high=100)}
svc_poly_param = { "kernel":"poly","class_weight":"balanced", "verbose":0}#, "cache_size":500}

best, raw = bhs.hparams_search(data, 'diagnosis', SVC(), svc_poly, svc_poly_param, scaler='MM', test_size= .2, cv=StratifiedKFold(5, shuffle=True), n_iter=100)
print('All Done!')

# Save poly params 
raw.to_csv(path_hparams + "/SVC_Poly_hparams.csv")

## Lightgbm - params
lgbm_grid ={#     'boosting_type': Categorical(['gbdt', 'dart', 'goss', 'rf']),
    'num_leaves': Integer(low=5, high=16),
    'max_depth': Integer(low=2, high=16),
    'learning_rate': Real(low=0.001, high=.3),
    'n_estimators': Integer(low=50, high=250),
    'reg_alpha':Real(low=0.1, high=.9),
    'reg_lambda':Real(low=0.1, high=.9)}

lgbm_param = {    'subsample':1.0,
    'subsample_freq':-1,
    'objective':'binary',
#     'early_stopping_round':50,
    'metric':'auc',
#     'class_weight': 'balanced',
    'n_jobs': -1,
    'verbose':-1#,
#     'categorical_feature': cat_cols
}

best, raw = bhs.hparams_search_lgbm(data, 'diagnosis', lgbm_grid, lgbm_param, scaler='none', test_size= .2, cv=StratifiedKFold(5, shuffle=True), n_iter=100)
print('All Done!')

# Save Lightgbm params 
raw.to_csv(path_hparams + "/lgbm_hparams.csv")

## XGBoost - params
xgb_grid = {
    'booster': Categorical(['gbtree', 'dart']),
    'tree_method': [ 'approx', 'hist'],
    'max_leaves': Integer(low=5, high=14),
    'max_depth': Integer(low=2, high=8),
    'max_bin': Integer(low=5, high=14),
    'learning_rate': Real(low=0.001, high=.3),
    'n_estimators': Integer(low=50, high=500),
    'reg_alpha':Real(low=0.1, high=.5),
    'reg_lambda':Real(low=0.1, high=.5)
}

xgb_param = {
    'gamma': 0.005,
    'subsample':1.0,
    'enable_categorical':True, # Supported tree methods are `gpu_hist`, `approx`, and `hist`.
    'n_jobs': -1,
    'verbosity':0,
    'eval_metric':'auc',
    'objective':'binary:logistic',
    'use_label_encoder':None
}

best, raw = bhs.hparams_search_xgb(data, 'diagnosis', xgb_grid, xgb_param, scaler='none', test_size= .2, cv=StratifiedKFold(5, shuffle=True), n_iter=100)
print('All Done!')

# Save XGBoost params 
raw.to_csv(path_hparams + "/xgb_hparams.csv")
print("##### ALL DONE! ##### ##### ALL DONE! ##### ##### ALL DONE! #####")
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
os.chdir('/home/marcelo/GitRepos/Tesis/code')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None



df = pd.read_csv("../data/clean_data.csv")

df.isna().sum()
without_gencog_data = df.loc[(df['mmse_total'].isna()) & (df['moca_total'].isna()) & (df['aceiii_total'].isna())].index

### Tiramos los casos que no tienen datos en cognición general
df = df.drop(without_gencog_data, axis=0)
df.site.value_counts()
df.isna().sum()

# Tiramos las variables que no sirven
df = df.drop(['aod', 'yod', 'laterality', 'income_sources', 'nationality', 'country_of_residence'], axis=1)


df.isna().sum()[df.isna().sum()>0].sort_values().sort_values()/df.shape[0]

# Vars to test imputation strategies using complete set for that var

cols =['years_education', 'year_birth', 'mmse_total', 'cdr_global',
       'functionality', 'cdr_sumofboxes', 'ifs_total_score', 'moca_total']
# count=0
# for j in tqdm(range(len(cols)), desc=f'Testing imputers for {cols[count]}', total=len(cols), colour='green'):
#     f


def test_iterative_imputer(data, nan_column, exclude=['none'], params='default',  scaler='MM',
                            n_iter=1000, nan_size=.1):
    """
    Description
    --------------------------------------------------------
    Implementation of iterative imputers for testing performance in one column. It takes a complete dataframe
    and generates NaNs for testing prurposeuse in a permutation  style
    Supported estimators are:
    Bayesian Ridge: 'BR'
    DecisionTreeRegressor: 'DTR'
    ExtraTreesRegressor: 'ETR'
    KNeighborsRegressor: 'KNR'
    
    Parameters
    --------------------------------------------------------
    data (Pandas Dataframe): Full df with X and y
    nan_column (str): Column name to generate nan and test iterative imputers
    exclude (list-lije): List with algotithms to exclude
    params (not implemented):###############
    scaler (str): sklearn scaler "MM" (MinMaxScaler), "SS" (StandardScaler), 'box-cox' or 'yeo-johnson' (PowerTransformer)
    n_iter: iterations for the bootstrap
    nan_size (float): float between 1 and 0. Proportion of NaNs to generate in the nan_columns
    
    Returns
    ----------------------------------------------------------
    Two pandas dataframe (mean and raw results)
    with mean RMSE, MSE and time in minutes per imputer and iteration
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, FunctionTransformer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import BayesianRidge
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import time
    
    data = data.reset_index(drop=True).copy()
    rows= data.index
    nan_size_n = int(rows[-1] * nan_size)
    num_cols=data.select_dtypes(include=[int, float]).columns
    
    # Scaler
    scaler_dict = {'MM': MinMaxScaler(feature_range=(0.05, 0.95)),
                   'SD': StandardScaler(),
                   'box-cox': PowerTransformer(method='box-cox'),
                   'yeo-johnson': PowerTransformer(method='yeo-johnson'),
                   'none': FunctionTransformer(lambda x: x)}
    scaler = scaler_dict.get(scaler, FunctionTransformer(lambda x: x))
    print(scaler)
    
    data.loc[:, num_cols] = scaler.fit_transform(data.loc[:, num_cols])
    
    # imputer
    imputer_dict = {'BR': BayesianRidge(n_iter=1000),
                'DTR': DecisionTreeRegressor(max_depth=int(len(data.columns)/4),min_samples_leaf=0.01),
                'ETR': ExtraTreesRegressor( max_depth=int(len(data.columns)/4),min_samples_leaf=0.01, n_jobs=-1),
                'KNR': KNeighborsRegressor(n_jobs=-1)}
    
    # Excluimos los que están en la lista exclude (parámetro)
    imputer_dict = {k: v for k, v in imputer_dict.items() if k not in exclude}
        
    ### Select indexes to save y_true and generate nans one column at time
    ## Then prdict and evaluate the imputer with RMSE and MSE
    res_cols = []
    for metric in ["_RMSE", "_MSE", "_Time"]:
        res_cols.extend(f"{j}{metric}" for j in imputer_dict.keys())
        
    results = pd.DataFrame(columns=res_cols, index=np.arange(n_iter))
    
    for i in tqdm(range(n_iter), desc='Iterating', leave=True, total=n_iter, colour='green'):
        selected_rows = np.random.choice(rows, size=nan_size_n, replace=False)
        y_true= data.loc[selected_rows,nan_column]
        
        
        for key, imputer in imputer_dict.items():
            data_nan = data.copy()
            data_nan.loc[selected_rows, nan_column] = np.nan
            start = time.time()
            imp = IterativeImputer(estimator=imputer)
            data_nan.loc[:, num_cols] = imp.fit_transform(data_nan.loc[:, num_cols])
            t = time.time()
            t = (t - start)/60
            y_pred = data_nan.loc[selected_rows, nan_column]
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            results.loc[i, f'{key}_RMSE'] = mse
            results.loc[i, f'{key}_MSE'] = rmse
            results.loc[i, f'{key}_Time'] = t
            
#             data_nan.loc[selected_rows, nan_column] = data.loc[selected_rows, nan_column].copy()
            
    mean_results = pd.DataFrame(results.mean()).T
    
    return mean_results, results

df.columns
df.loc[df['years_education']=='.'] = np.nan
df.years_education = df.years_education.astype('float32')

df.isna().sum().sort_values().sort_values()/df.shape[0]

df.diagnosis.unique()
diagnosis = pd.get_dummies(df.diagnosis).drop('CN', axis=1)
df = pd.concat([df, diagnosis], axis=1)

df = df[['AD', 'FTD', 'sex',  'years_education', 'year_birth', 'cognition', 'cdr_global',
       'functionality', 'ifs_total_score', 'ifs_total_score', 'mmse_vs', 'moca_vs', 'ace_vs']]

test_df = df.dropna().copy()

cols =['years_education', 'year_birth', 'cognition', 'cdr_global',
       'functionality', 'ifs_total_score', 'ifs_total_score', 'mmse_vs', 'moca_vs', 'ace_vs']


res = {i:[] for i in cols}
res
for col in range(len(cols)):
    m, _ = test_iterative_imputer(test_df, cols[col], nan_size=.8 )
    res[cols[col]] = m



res_df = pd.DataFrame()

for i in res.keys():
    res_df= pd.concat([res_df, res[i]], axis=0, ignore_index=True)

res_df['Vars'] = list(res.keys())
res_df = res_df.set_index('Vars')
res_df.to_csv("../imp_results/Iterative_imputer_test_80percent_nans_.csv")

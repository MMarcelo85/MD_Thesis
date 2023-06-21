import numpy as np
import pandas as pd
from tqdm import tqdm
import os
os.chdir('/home/marcelo/GitRepos/Tesis/code')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

import miceforest as mf
import time
from sklearn.metrics import mean_squared_error



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

df.columns
df.loc[df['years_education']=='.'] = np.nan
df.years_education = df.years_education.astype('float32')

df.isna().sum().sort_values().sort_values()/df.shape[0]

df.diagnosis.unique()
diagnosis = pd.get_dummies(df.diagnosis).drop('CN', axis=1)
df = pd.concat([df, diagnosis], axis=1)

df = df[['AD', 'FTD', 'sex',  'years_education', 'year_birth', 'cognition', 'cdr_global',
       'functionality', 'ifs_total_score', 'mmse_vs', 'moca_vs', 'ace_vs']]

test_df = df.dropna().copy()

cols =['years_education', 'year_birth', 'cognition', 'cdr_global',
       'functionality', 'ifs_total_score', 'mmse_vs', 'moca_vs', 'ace_vs']



def test_iterative_MICE(data, nan_column, scaler='MM', n_iter=1000, nan_size=.1):
    
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, FunctionTransformer
    from sklearn.metrics import mean_squared_error
    import miceforest as mf
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
    
    data.loc[:, num_cols] = scaler.fit_transform(data.loc[:, num_cols])
    
    # n datasets for imputer
    n_datasets = [3,5,7]
        
    ### Select indexes to save y_true and generate nans one column at time
    ## Then prdict and evaluate the imputer with RMSE and MSE
    res_cols = []
    for metric in ["_dfs_RMSE", "_dfs_MSE", "_dfs_Time"]:
        res_cols.extend(f"{j}{metric}" for j in n_datasets)
        
    results = pd.DataFrame(columns=res_cols, index=np.arange(n_iter))
    
    for i in tqdm(range(n_iter), desc='Iterating', leave=True, total=n_iter, colour='green'):
        selected_rows = np.random.choice(rows, size=nan_size_n, replace=False)
        y_true= data.loc[selected_rows,nan_column]
        
        
        for n in range(len(n_datasets)):
            data_nan = data.copy()
            data_nan.loc[selected_rows, nan_column] = np.nan
            for columna in test_df.columns[2:]:
                 test_df[columna] = test_df[columna].astype('float32')

            start = time.time()
            # Create kernels. 
            kernel = mf.ImputationKernel(data_nan, datasets=n_datasets[n], save_all_iterations=True, random_state=123)
           # Run MICE
            kernel.mice(n_datasets[n]-1)
            imputed_dataset = kernel.complete_data(dataset=n_datasets[n]-1)

            t = time.time()
            t = (t - start)/60
            y_pred = imputed_dataset.loc[selected_rows, nan_column]
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            results.loc[i, f'{str(n_datasets[n])}_dfs_RMSE'] = mse
            results.loc[i, f'{str(n_datasets[n])}_dfs_MSE'] = rmse
            results.loc[i, f'{str(n_datasets[n])}_dfs_Time'] = t
            
            data_nan.loc[selected_rows, nan_column] = data.loc[selected_rows, nan_column].copy()
            
    mean_results = pd.DataFrame(results.mean()).T
    
    return mean_results, results


df = df[['AD', 'FTD', 'sex',  'years_education', 'year_birth', 'cognition', 'cdr_global',
       'functionality', 'ifs_total_score',  'mmse_vs', 'moca_vs', 'ace_vs']]

test_df = df.dropna().copy()

cols =['years_education', 'year_birth', 'cognition', 'cdr_global',
       'functionality', 'ifs_total_score', 'mmse_vs', 'moca_vs', 'ace_vs']

res = {i:[] for i in cols}
res
for col in range(len(cols)):
    print(cols[col])
    m, _ = test_iterative_MICE(test_df, cols[col], nan_size=.8 )
    res[cols[col]] = m



res_df = pd.DataFrame()
for i in res.keys():
    res_df= pd.concat([res_df, res[i]], axis=0, ignore_index=True)

res_df['Vars'] = list(res.keys())
res_df = res_df.set_index('Vars')
res_df.to_csv("../results/MICE_test_80percent_nans.csv")


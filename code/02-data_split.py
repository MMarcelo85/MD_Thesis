import numpy as np
import pandas as pd
import os
os.getcwd()
os.chdir("/home/marcelo/GitRepos/Tesis/code")
import py_funs
import stats_funs
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None
# Data path
data_path ="/home/marcelo/GitRepos/Tesis/data/"
# Load data 
df = pd.read_csv(data_path+'clean_data.csv')
# Drop participants with no MMSE or MoCA or ACE-III
df = df.dropna(subset='mmse_vs')
df.isna().sum()
df.shape # 1613, 48
df = df.drop(['aod', 'yod', 'laterality', 'income_sources' ], axis=1)
df['years_education']=df['years_education'].astype('float')
df['nationality'].unique()
nat_dictionary = {'Argentina':1, 'Chilena':2, 'Colombiana': 3, 'Mexicana':4, 'PERUANA':5, 'Española':6, 'Alemana':7}
# Tiramos un espaqñol y un alemán
df = df.loc[(df['nationality']!='Española') & (df['nationality']!='Alemana')]
# Reemplazamos
df['nationality'] = df['nationality'].replace(nat_dictionary)

# Idem con Residencia
df['country_of_residence'].unique()
res_dictionary = {'Argentina':1, 'Chile':2, 'Colombia': 3, 'Mexico':4, 'Peru':5, 'Panamá':6}
df['country_of_residence'] = df['country_of_residence'].replace(res_dictionary)

# Salvamos el df a modelar
df.to_csv(data_path+"data_to_model.csv", index=False)
df.info()

####   Separate df between train and test (25%)
# Make strata column
df['strata'] = df.diagnosis + " / " + df.site

test = py_funs.stratified_sample(df, ['strata'], size=int(df.shape[0]*.10), seed=123, keep_index=True)
test = test.set_index('index')
test.index.name = None
train = df.drop(test.index, axis=0)
train.shape, test.shape

df.diagnosis.value_counts()/df.shape[0]
train.diagnosis.value_counts()/train.shape[0]
test.diagnosis.value_counts()/test.shape[0]

# final_cols = ['site', 'id', 'diagnosis', 'year_birth', 'sex', 'years_education','cognition', 'functionality','ifs_total_score', 'mini_sea_total', 'npi_total', 'npi_total_caregiver', 'marital_status', 'n_children', 'household_members', 'household_income', 'Job_status', 'income_s_NaN', 'income_s_1', 'income_s_2', 'income_s_3', 'income_s_4', 'income_s_5', 'income_s_6', 'income_s_7', 'income_s_8', 'income_s_9', 'income_s_10', 'income_s_11', 'strata']

# train = train[final_cols]
# test = test[final_cols]
#save train and test 
train.to_csv("../data/train.csv", index=False)
test.to_csv("../data/test.csv", index=False)

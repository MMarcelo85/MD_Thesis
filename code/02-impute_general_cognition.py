import pandas as pd
import numpy as np

from tqdm import tqdm
import os
os.chdir('/home/marcelo/GitRepos/Tesis/code')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

import GC_transform

data_path = '/home/marcelo/GitRepos/Tesis/data/'
df = pd.read_csv(data_path + 'clean_data.csv')
df = df.dropna(subset=['age', 'years_education']).reset_index()
df.head()

df.isna().sum()[df.isna().sum()>0]

df.loc[(df['mmse_total'].notna() ) | (df['moca_total'].notna() ) | (df['aceiii_total'].notna() ) ].shape

# Transformación e imputación
## MMSE
### Valores para MMSE con MoCA y ACE-III
df['mmse_vs'] = np.nan
df['mmse_lw'] = np.nan

mmse_vs = GC_transform.MoCa_to_MMSE(df['moca_total'], method='Van Steenoven')
mmse_lw = GC_transform.MoCa_to_MMSE(df['moca_total'], method='Lawton')
mmse_mg = GC_transform.ACEIII_to_MMSE(df['aceiii_total'])

df['mmse_vs'] = np.where(df['mmse_total'].isna(), mmse_vs, df['mmse_total'] )
df['mmse_vs'] = np.where(df['mmse_vs'].isna(), mmse_mg, df['mmse_vs'] )
df['mmse_lw'] = np.where(df['mmse_total'].isna(), mmse_lw, df['mmse_total'] )
df['mmse_lw'] = np.where(df['mmse_lw'].isna(), mmse_mg, df['mmse_lw'] )

# Sanity Check (ok)
# assert(df.loc[(df['mmse_vs'].isna()) & ( (df['mmse_total'].notna()) | (df['moca_total'].notna()) | (df['aceiii_total'].notna()) ), ['mmse_vs', 'mmse_lw', 'mmse_total', 'moca_total', 'aceiii_total']].shape[0]==0)

## MoCA
### Valores para MoCA con MMSE (usando las variables imputadas que tienen 1- Valores originales de MMSE, 2- Valores imputados con VS o Lw, 3- Valores imputados con ACE-III y 4- NaN para aquellos que no tienen ninguna de las tres pruebas)

df['moca_vs'] = np.nan
df['moca_lw'] = np.nan

moca_vs = GC_transform.MMSE_to_MoCA(df['mmse_vs'])
moca_lw = GC_transform.MMSE_to_MoCA(df['mmse_lw'])

df['moca_vs'] = np.where(df['moca_total'].isna(), moca_vs, df['moca_total'])
df['moca_lw'] =  np.where(df['moca_total'].isna(), moca_lw, df['moca_total'])

# Sanity Check (ok)
# assert(df.loc[(df['mmse_vs'].isna()) & ( (df['mmse_total'].notna()) | (df['moca_total'].notna()) | (df['aceiii_total'].notna())| (df['moca_vs'].notna())| (df['moca_lw'].notna()) ), ['mmse_vs', 'mmse_lw', 'mmse_total', 'moca_total', 'aceiii_total', 'moca_vs', 'moca_lw']].shape[0]==0)

## ACE-III
### Idem anterior pero para ACE-III con MMSE
df['ace_vs'] = np.nan
df['ace_lw'] = np.nan

ace_vs = GC_transform.MMSE_to_ACEIII(df['mmse_vs'])
ace_lw = GC_transform.MMSE_to_ACEIII(df['mmse_lw'])

df['ace_vs'] = np.where(df['aceiii_total'].isna(), ace_vs, df['aceiii_total'])
df['ace_lw'] = np.where(df['aceiii_total'].isna(), ace_lw, df['aceiii_total'])


df.isna().sum()[df.isna().sum()>0]
df.info()

df.income_sources.unique()

import pandas as pd
import numpy as np

str(list((range(1, 12))))

# Crear DataFrame con la columna
income = pd.DataFrame( columns=['NaN'] + list((range(1, 12))), index=range(len(df)))
income.columns = [str(col) for col in income.columns]
income.columns
income = income.replace({np.nan:0})
income
# Convertir la columna en una lista separada por comas de los valores únicos
respuestas = df['income_sources']

respuestas = respuestas.replace({"1.0":"1", "5.0": "5", "8.0":"8","9.0": "9", np.nan:"NaN"})
respuestas.unique()

for i in range(len(respuestas)):
    # print(i)
    respuesta = respuestas[i].split(",")
    for l in range(len(respuesta)):
        item = respuesta[l]
        # print(i, item)
        item = item.strip()
        if item in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "NaN"]:
            income.loc[i, item] = 1



income['respuestas'] = respuestas

income = income.iloc[:, :-1]
income.columns = [("income_s_"+i) for i in ['NaN', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']]
[("income_s_"+i) for i in ['NaN', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']]

assert(income.shape[0] == df.shape[0])
df =pd.concat([df, income], axis=1)

# df = df[['site', 'id', 'diagnosis', 'year_birth', 'sex', 'aod', 'yod', 'years_education', 'laterality', 'moca_total', 'aceiii_total', 'mmse_total', 'ifs_total_score', 'mini_sea_total', 'pfeffer_total', 'cdr_sumofboxes', 'cdr_global', 'npi_total', 'npi_total_caregiver', 'nationality', 'country_of_residence', 'marital_status', 'n_children', 'household_members', 'household_income', 'income_sources', 'Job_status', 'mmse_vs', 'mmse_lw', 'moca_vs', 'moca_lw', 'ace_vs', 'ace_lw', 'income_s_NaN', 'income_s_1', 'income_s_2', 'income_s_3', 'income_s_4', 'income_s_5', 'income_s_6', 'income_s_7', 'income_s_8', 'income_s_9', 'income_s_10', 'income_s_11']]

df.to_csv("/home/marcelo/GitRepos/Tesis/data/clean_data.csv",index=False)
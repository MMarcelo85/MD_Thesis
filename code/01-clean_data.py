import pandas as pd
import numpy as np

from tqdm import tqdm
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

data_path = '/home/marcelo/MEGA/RedLat/2022/Pre-existing/raw/PRE-EXISTING/'
data_path2 = '/home/marcelo/MEGA/RedLat/2022/Pre-existing/raw/PRE-EXISTING SDH SES DATA/'

files = [file for file in os.listdir(data_path) if file.endswith('xlsx')]
f_cols= ['Site', 'id', 'diagnosis','gender', 'year_of_birth','age_onset_disease', 'year_of_diagnosis', 'years_education', 'laterality', 'moca_total',  'aceiii_total', 'mmse_total', 'ifs_total_score', 'mini_sea_total', 'pfeffer_total', 'cdr_sumofboxes', 'cdr_global', 'npi_total', 'npi_total_caregiver']


df = pd.DataFrame()

columnas = pd.DataFrame(pd.read_excel(data_path + 'prior_data_per_subject_ib.xlsx', skiprows=1))

for i in tqdm(files):
    print(i)
    temp_df = pd.read_excel(data_path + i, skiprows=1)
    test = temp_df.columns.equals(columnas.columns)
    print(test)
    if test !=True:
        bien = columnas.columns
        mal = temp_df.columns
        for i in range(len(bien)):
            if bien[i] != mal[i]:
                print(f"Nombre correcto: {bien[i]} ---------->Nombre incorrecto: {mal[i]}\n")
                temp_df.columns = columnas.columns
            else:
                pass
        
        temp_df.columns = columnas.columns
        print('corregido!')
    else:
        pass    
    df= pd.concat([df, temp_df], axis=0, ignore_index=True)

df = df[f_cols]

df2 = pd.DataFrame()

files = [file for file in os.listdir(data_path2) if file.endswith('xlsx')]
for i in tqdm(files):
    temp_df = pd.read_excel(data_path2 + i, skiprows=1, usecols=['id', 'Site', 'nationality', 'country_of_residence', 'education_years', 'marital_status', 'n_children', 'household_members', 'household_income', 'income_sources', 'Job_status'])
    df2= pd.concat([df2, temp_df], axis=0, ignore_index=True)

df = df.merge(df2, on=['id', 'Site'], how='left')
df = df.drop(['diagnosis_y', 'gender_y'], axis=1)
df.shape
df.isna().sum()[df.isna().sum()>0]

#### Clean
df = df.dropna(subset=['diagnosis', 'gender'])
df.loc[(df['years_education'].isna()) & (df['education_years'].notna())]
df['years_education'] = np.where((df['years_education'].isna()) & (df['education_years'].notna()), df['education_years'], df['years_education'])
df['diagnosis'] = df['diagnosis'].str.strip()
df = df.loc[df['Site']!='Takada'] # Brazil has no data
# df['age'] = 2023 -  df['year_of_birth']

df.isna().sum()[df.isna().sum()>0].sort_values()
df.Site.replace({'SOSA':'Sosa', 'Avila Funes': 'Avila'}, inplace=True)
df.Site.unique()


def impute_territory(dataframe, variable):
    dataframe.loc[(dataframe[variable].isna()) & (dataframe['Site'].isin(['Matallana', 'Lopera', 'Cardona'])), variable] = 'Colombia'
    dataframe.loc[(dataframe[variable].isna()) & (dataframe['Site'].isin(['Ibanez', 'Bruno', 'Brusco'])), variable] = 'Argentina'
    dataframe.loc[(dataframe[variable].isna()) & (dataframe['Site'].isin(['Behrens', 'Slachevsky'])), variable] = 'Chile'
    dataframe.loc[(dataframe[variable].isna()) & (dataframe['Site'].isin(['Avila', 'Sosa'])), variable] = 'Mexico'
    dataframe.loc[(dataframe[variable].isna()) & (dataframe['Site'].isin(['Custodio'])), variable] = 'Peru'
    return dataframe
df = impute_territory(df, 'nationality')
df = impute_territory(df, 'country_of_residence')
df['country_of_residence'].replace({'PERÚ':'Peru', 'chile': 'Chile'}, inplace = True)
df['nationality'].replace({'Peruana': 'Peruana', 'ESPAÑOLA': 'Española', 'ALEMANA': 'Alemana', 'Chile': 'Chilena', 'chile': 'Chilena', 'Colombia': 'Colombiana', 'Mexico': 'Mexicana', 'argentina': 'Argentina', 'Ecuador': 'Ecuatoriana'}, inplace=True)

df = df.drop(['education_years'], axis=1)
df.isna().sum()[df.isna().sum()>0].sort_values()
df.columns
df = df[['Site', 'id', 'diagnosis', 'year_of_birth', 'gender', 'age_onset_disease','year_of_diagnosis', 'years_education', 'laterality', 'moca_total', 'aceiii_total', 'mmse_total', 'ifs_total_score', 'mini_sea_total', 'pfeffer_total', 'cdr_sumofboxes', 'cdr_global', 'npi_total', 'npi_total_caregiver', 'nationality', 'country_of_residence', 'marital_status', 'n_children', 'household_members','household_income', 'income_sources', 'Job_status']]

df.columns=['site', 'id', 'diagnosis', 'year_birth', 'sex', 'aod','yod', 'years_education', 'laterality', 'moca_total','aceiii_total', 'mmse_total', 'ifs_total_score', 'mini_sea_total', 'pfeffer_total', 'cdr_sumofboxes', 'cdr_global', 'npi_total', 'npi_total_caregiver', 'nationality', 'country_of_residence', 'marital_status', 'n_children', 'household_members', 'household_income', 'income_sources', 'Job_status']

os.getcwd()
df.to_csv('./Tesis/data/clean_data.csv', index=False)

df.loc[(df['mmse_total'].isna() ) & (df['moca_total'].isna() ) & (df['aceiii_total'].isna() ) ]#.to_csv("~/Desktop/Pre.csv", index=False)

df.isna().sum()
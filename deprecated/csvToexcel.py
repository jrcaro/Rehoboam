import pandas as pd 

df1 = pd.read_csv('data/camaras_distr.csv', delimiter=';')
df2 = pd.read_csv('data/cameras_info.csv')

df = df2.set_index('camera_name').join(df1.set_index('Camara'))
df = df.drop(columns='Distrito').reset_index(drop=False)

df.to_excel('rehoboam_data.xlsx', sheet_name='cameras', index=False)
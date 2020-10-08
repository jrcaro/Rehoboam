import pandas as pd

if __name__ == "__main__":
    df_vehiculos = pd.read_excel("data/parque_2019_anuario.xlsx", sheet_name='V_4', skiprows=2, skipfooter=1)
    df_censo = pd.read_csv('data/censo_provincias.csv', sep=';', thousands='.')

    df_censo['Provincias'] = df_censo['Provincias'].map(lambda x: x[3:].split('/')[0])
    df_vehiculos['Provincias'.upper()] = df_vehiculos['Provincias'.upper()].map(lambda x: x.split('/')[0].replace(' (', ', ').replace(')', ''))

    df = df_vehiculos.set_index('PROVINCIAS').join(df_censo.set_index('Provincias'))
    df['Veh por habitante'] = df['TOTAL']/df['Total']*1000
    print(df.sort_values(by=['Veh por habitante'])['Veh por habitante'].loc['Malaga'])
import pandas as pd
import folium

'''df = pd.read_csv("data/da_camarasTrafico-4326.csv")
df_distr = pd.read_csv("data/camaras_distr.csv", delimiter=';')

df = df[['name', 'wkb_geometry']]

df = df_distr.set_index('Camara').join(df.set_index('name'))
df = df.reset_index(drop=False)\
        .rename(columns={'Camara': 'name', 'Distrito': 'id_district'})

name_id = df['name'].str.split(r'-', expand=True, n=1)
df['id_camera'] = name_id[0].str[2:]

coordinates = df['wkb_geometry'].str.extract(r'(-?\d{1,}.\d+ -?\d{1,}.\d+)')[0]\
                    .str.split(" ", expand=True)
df['lat'] = coordinates[1].astype(float)
df['lon'] = coordinates[0].astype(float)

df = df[['id_camera', 'id_district', 'name', 'lat', 'lon']].set_index('id_camera')
#df.to_csv('data/cameras_info.csv')'''

df = pd.read_csv('data/cameras_info.csv')
df = df.set_index('id_district', drop=True)
df = df[df['readable'] == 1]
df = df.dropna()

m = folium.Map(location=[36.716667, -4.416667], zoom_start=12)
for i in range(df.shape[0]):
    folium.Marker([df['lat'].iloc[i], df['lon'].iloc[i]], popup=df['camera_name'].iloc[i]).add_to(m)

m.save('data/cameras.html')
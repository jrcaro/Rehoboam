import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import base64
from utils import kafkaProducer, mongoConnect, hour_dict
import glob
from kafka.admin import KafkaAdminClient, NewTopic
from kafka import KafkaConsumer
import plotly.graph_objects as go
import pymongo
import numpy as np
from datetime import datetime
import os
import json
import time

# ==============================================================================
# Initialization
# ==============================================================================
#Config parameters for yolo
config = {
    'weights': './data/models/YOLO/yolov4-obj_6000.weights',
    'input_size': 416,
    'score_thres': 0.8,
    'model': 'yolov4',
    'weights_tf': 'data/models/YOLO/checkpoints/yolov4_imbalanced',
    'output_path': 'data/result.jpg',
    'iou': 0.45
}

#Remove the result image
if os.path.exists(config['output_path']):
    os.remove(config['output_path'])

#Name and inicialization of Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True,  external_stylesheets=[dbc.themes.BOOTSTRAP])
#Path to the excel data
path_data = 'data/rehoboam_data.xlsx'
#Path of the classes file
path_names = 'data/models/YOLO/obj.names'
#Path for about page
path_img = 'data/IMG-20190605-WA0003.jpg'
encoded_image_about = base64.b64encode(open(path_img, 'rb').read())

# Classes in the images
tag_df = pd.read_excel(path_data, sheet_name='classes')
tags = dict(zip(tag_df.name, tag_df.tag))
tags_rever = dict(zip(tag_df.tag, tag_df.name))

#Kafka client inicialization
admin_client = KafkaAdminClient(
    bootstrap_servers=['localhost:9094', 'localhost:9095'], 
    client_id='test'
)

#Input topic creation
input_topic = "input_image"
topic_list = []
consumer = KafkaConsumer(bootstrap_servers=['localhost:9094', 'localhost:9095'], group_id="rehoboam")
        
if input_topic not in consumer.topics():
    topic_list.append(NewTopic(name=input_topic, num_partitions=2, replication_factor=2))
    admin_client.create_topics(new_topics=topic_list, validate_only=False)

#Font for the html style
font = 'Arial'

#Connection to the Mongo collection to write in
mongo_col = mongoConnect()

#Read the class file and transform in dictionary
with open(path_names) as f:
    names_dict = {i: line.split('\n')[0] for i,line in enumerate(f)}

#Read the CSV file with the data of the cameras and transform it to Pandas dataframe
cameras_df = pd.read_excel(path_data, sheet_name='cameras')
cameras_df = cameras_df.set_index('id_district', drop=True)
cameras_df = cameras_df[cameras_df['readable'] == 1]

#Malaga's university logo
UMA_LOGO = "https://www.uma.es/servicio-comunicacion/navegador_de_ficheros/Marcas-UMA/descargar/Marca%20Universidad%20de%20M%C3%A1laga/marcauniversidaddemalagaVERTICAL.png"

# ==============================================================================
# Forms menus
# ==============================================================================

# make a reuseable navitem for the different examples
nav_items = dbc.Nav(
    [
        dbc.NavItem(dbc.NavLink("Tráfico en tiempo real", href="/real-traffic")),
        dbc.NavItem(dbc.NavLink("Histórico", href="/data-search")),
        dbc.NavItem(dbc.NavLink("Localización", href="/camera-location")),
        #dbc.NavItem(dbc.NavLink("Bitbucket", href="#")),
        dbc.NavItem(dbc.NavLink("Sobre mí", href="/about"))
    ], 
    className="ml-auto", 
    navbar=True, 
    style={
        'font-weight': 'bold',
        'font-size': '18px',
        }
)

# Logo to the navbar brand
logo = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.A(html.Img(src=UMA_LOGO, height="50px"), href='www.uma.es')),
                    ],
                    align="center",
                    no_gutters=False,
                ),
                href="/",
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(nav_items,
                id="navbar-collapse",
                navbar=True,
            ),
        ], fluid=True
    ),
    color="#81b7d6",
    dark=False
)

#Dropdown menu for Malaga's district
district_menu = html.Div(children=[
    dbc.FormGroup(
        [
            html.Label('Seleccione un distrito:',
            className='mr-3',
            style={
                'font-weight': 'bold',
                'color': 'black'
            },
            title='Seleccione uno de los distritos de la ciudad de Málaga.'),
            dcc.Dropdown(
                id='district-dropdown',
                options=[{'label': 'Distrito {}: {}'.format(j,i), 'value': j} for i,j in
                            zip(
                                cameras_df['district_name'].unique(),
                                cameras_df.index.unique()
                            )],
                value=1)
        ]
    )
])

#Dropdown menu for the cameras
cameras_menu = html.Div(children=[
    dbc.FormGroup([
        html.Label('Seleccione la camara:',
        className='mr-3',
            style={
                'font-weight': 'bold',
                'color': 'black'
            },
        title='Seleccione la cámara que desea ver. En el botón\n'
            '\"Localización\" de la barra de navegación puede ver\n'
            'la posición de cada una de las cámaras.'),
        dcc.Dropdown(
            id='camera-dropdown'
        )
    ])
])

#Radio items for the style of the bar chart
bar_chart_control = html.Div(children=[
    dbc.FormGroup([
        html.Label('Seleccione una opción:',
        className='mr-3',
        style={
            'font-weight': 'bold',
            'color': 'black'
        },
        title='Estas opciones representan la forma de agrupar\n'
            'el diagrama de barras. Si se agrupan por dirección\n'
            'se verán 16 clases, uno por cada dirección. Si se\n'
            'agrupa por vehículo se sumaran las frecuencia de las\n'
            'direcciones de cada tipo de vehículo.'),
        dcc.RadioItems(id='radio_buttons_id',
            options=[
                {'label': ' Mostrar por dirección', 'value': 1, 'disabled': True},
                {'label': ' Mostrar por tipo de vehículo', 'value': 2, 'disabled': True}
            ],
            value=1,
        )
    ])
])

#Date picker menu for the date to display in line chart
date_picker_menu = html.Div(children=[
    dbc.FormGroup([
        html.Label('Seleccione la fecha:',
        className='mr-3',
        style={
            'font-weight': 'bold',
            'color': 'black'
        },
        title='Fecha de la que desee extraer los datos. Sólo\n'
            'existen datos desde el 17 de Septiembre, fecha de\n'
            'inicio del proyecto.'),
        html.Br(),
        dcc.DatePickerSingle(
            id='date-picker',
            min_date_allowed=datetime(2020,9,1),
            first_day_of_week=1,
            display_format='DD/MM/YYYY',
            placeholder='Fecha'
        ) ,
    ])
])

#Dropdown class selection to display in line chart
class_menu = html.Div(children=[
    dbc.FormGroup([
        html.Label('Seleccione las clases:',
        className='mr-3',
        style={
            'font-weight': 'bold',
            'color': 'black'
        },
        title='Representan las 16 clases existentes. Es posible\n'
            'escoger más de una y se irán añadiendo al gráfico.'),
        dcc.Dropdown(
            id='class-dropdown',
            value='Coche Frontal',
            options=[{'label': k, 'value': v} for k,v in tags.items()],
            multi=True
        )
    ])
])

#Slider to select the hours to display in line chart
slider_group = html.Div(id='div-slider', children=[
    dbc.FormGroup([
        html.Label('Seleccione un rango de horas:',
        className='mr-3',
        style={
            'font-weight': 'bold',
            'color': 'black'
        },
        title='Este selector permite filtrar la gráfica entre\n'
            'el rango de horas que se desee. Si desea realizar\n'
            'un zoom más específico, puede arrastrar el cursor\n'
            'sobre la parte de la gráfica que desee.'),
        dcc.RangeSlider(
            id='hour-slider',
            min=0,
            max=23,
            step=None,
            marks={hour: 
                {'label': str(hour), 'style': {'font-size': '15px'}
                } for hour in range(0,24)},
            value=[0, 23],
            pushable=1)
    ])
], hidden=True)

#Image of the detection result
image_html = html.Img(
    id='image_id',
    style={'height': '80%', 'width':'80%'})

# ==============================================================================
# Layout content
# ==============================================================================
#Layout for the real time traffic page
layout_real_time = html.Div([
    dcc.Interval(id='interval_id', interval=5*1000),   
    dbc.Row(html.H1("Análisis del tráfico en tiempo real".upper()),
        justify="center",
        align="center",
        className="h-50", 
        style={
            'padding-top': '10px',
            'padding-bottom': '40px'}),
    dbc.Row([
        dbc.Col(district_menu, width=3),
        dbc.Col(cameras_menu, width=3),
    ]),
    dbc.Row(dbc.Col(children=[
        bar_chart_control
    ], width=2)),
    dbc.Row([
        dbc.Col(image_html, width={'size': 6}, align='center'),
        dbc.Col(id='bar_chart', width={'size': 6}, align='center')
    ], justify="center"),
    html.P(id='placeholder'),
    #html.Div(id="debug") #OJO
], className="mx-3")

#Layout for the data search page
layout_search = html.Div([
    dbc.Row(html.H1("Histórico de los datos".upper()),
        justify="center",
        align="center",
        className="h-50", 
        style={
            'padding-top': '10px',
            'padding-bottom': '40px'}),
    dbc.Row([
        dbc.Col(district_menu, width=3),
        dbc.Col(cameras_menu, width=3)
    ]),
    dbc.Row([
        dbc.Col(class_menu, width=3),
        dbc.Col(date_picker_menu, width=3)
    ]),
    dbc.Row(dbc.Col(slider_group, width=6)),
    dbc.Spinner(html.Div(id='line-chart', style={'width': '100%'}),
                #type="grow",
                color="#81b7d6",
                spinner_style={"width": "4rem", "height": "4rem"})
], className="mx-3")

#Layout for the cameras location page
layout_location = html.Div([
    dbc.Row(html.H1("Localización de las cámaras".upper()),
        justify="center",
        align="center",
        className="h-50", 
        style={
            'padding-top': '10px',
            'padding-bottom': '40px'}),
    html.Iframe(id='map', srcDoc=open('data/cameras.html', 'r').read(), width='100%', height='500')
], className="mx-3")

#Layout for the about page
layout_about = html.Div([
    dbc.Row(html.H1("Sobre mí".upper()),
        justify="center",
        align="center",
        className="h-50", 
        style={
            'padding-top': '10px',
            'padding-bottom': '10px'}),
    dbc.Row(
        html.Img(
            src='data:image/png;base64,{}'.format(encoded_image_about.decode()),
            style={'width': '300px', 'height': '300px'}
        ), align="center", justify="center"
    ),
    html.Br(),
    html.Iframe(
        id='about', 
        srcDoc=open('data/about_me.html', 'r').read(), 
        width='100%', 
        height='550',
        style={
            'font-family': font
        })
], className="mx-3")

# ==============================================================================
# Main layout
# ==============================================================================
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    logo,
    html.Div(id='page-content'),
    html.Footer('Creado por Juan Rafael Caro Romero', style={
        'position': 'absolute',
        'bottom': '0',
        'width': '100%',
        'height': '20px',
        'font-style': 'italic',
        'color': '#c6cdd6'
    }, className='mx-3')
], style={
    'width':'100%', 
    'height':'100%', 
    'position': 'absolute',
    'font-family': font
    }
)

# ==============================================================================
# Callbacks
# ==============================================================================
# Callback for the page navigation
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/real-traffic' or pathname == '/':
        return layout_real_time
    elif pathname == "/data-search":
        return layout_search
    elif pathname == "/camera-location":
        return layout_location
    elif pathname == "/about":
        return layout_about


# Callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

#Callback for the addition of the camera dropdown
@app.callback(
    Output('camera-dropdown', 'options'),
    [Input('district-dropdown', 'value')])
def update_output(value):
    return [{'label': k['camera_name'], 'value': k['id_camera']} for k in
                cameras_df.loc[value][['id_camera', 'camera_name']].to_dict('records')]

#Show the bar chart
@app.callback(
    Output('bar_chart', 'children'),
    [Input('interval_id', 'n_intervals'),
    Input('district-dropdown', 'value'),
    Input('camera-dropdown', 'value'),
    Input('radio_buttons_id', 'value')]
)
def update_bar_chart(n, district_id, camera_id, value):
    if camera_id != None and district_id != None:
        data = mongo_col.find_one({'$and': [{'camera_id': camera_id}, {'district_id': district_id}]}, sort=[('timestamp', pymongo.DESCENDING)])
        fig = go.Figure()

        if value == 1 and data != None:  
            #fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[i for i in sorted(tags.keys(), key=lambda x: x)],
                y=list(data['results'].values()),
                text=list(data['results'].values()),
                textposition='auto',
                name='Primary Product',
                marker_color='indianred'
            ))
        elif value == 2:
            count = [0]*4
            for class_, num in data['results'].items():
                if class_.find('coche') != -1:
                    count[2] += num
                elif class_.find('camion') != -1:
                    count[1] += num
                elif class_.find('moto') != -1:
                    count[3] += num
                elif class_.find('autobus') != -1:
                    count[0] += num
                
            #fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Autobús', 'Camión', 'Coche', 'Motocicleta'],
                y=count,
                text=count,
                textposition='auto',
                name='Primary Product',
                marker_color='indianred'
            ))
        fig.update_layout(
            xaxis_tickangle=315,
            autosize=True,
            height=500,
            font=dict(
                family=font,
                size=15
            ),
            yaxis=dict(
                title='Frecuencia',
                tickmode='linear',
                tick0=0,
                dtick=1
            )
        )

        return dcc.Graph(figure=fig)

#Enable the radio items for the bar chart
@app.callback(
    Output('radio_buttons_id', 'options'),
    [Input('camera-dropdown', 'value')]
)
def enable_radio(value):
    if value != None:
        options = [
            {'label': ' Mostrar por dirección', 'value': 1, 'disabled': False},
            {'label': ' Mostrar por tipo de vehículo', 'value': 2, 'disabled': False}
        ]
    else:
        options = [
            {'label': ' Mostrar por dirección', 'value': 1, 'disabled': True},
            {'label': ' Mostrar por tipo de vehículo', 'value': 2, 'disabled': True}
        ]
    
    return options

#Callback for the image load
@app.callback(
    Output('image_id', 'src'),
    [Input('camera-dropdown', 'value'),
    Input('interval_id', 'n_intervals')]
)
def load_image(value, n):
    if value != None and os.path.exists('./' + config['output_path']):
        encoded_image = base64.b64encode(open('./' + config['output_path'], 'rb').read())
        return 'data:image/png;base64,{}'.format(encoded_image.decode())

#Callback for lauch the kafka producer
@app.callback(
    Output('placeholder', 'children'),
    [Input('camera-dropdown', 'value'),
    Input('district-dropdown', 'value'),
    Input('interval_id', 'n_intervals')]
)
def init_producer(value_cam, value_dist, n):
    if value_cam != None and value_dist != None:
        kafkaProducer(camera_id=value_cam, district_id=value_dist)

#Create the date picker from the last data available
@app.callback(
    Output('date-picker', 'max_date_allowed'),
    [Input('district-dropdown', 'value'),
    Input('camera-dropdown', 'value')]
)
def search_data(district_id, camera_id):
    if district_id != None and camera_id != None:
        data = mongo_col.find({'$and': [{'camera_id': camera_id}, {'district_id': district_id}]}, sort=[('timestamp', pymongo.DESCENDING)])
        time_list = [time['timestamp'] for time in data]

        if time_list != []:
            max_date = datetime(time_list[0].year, time_list[0].month, time_list[0].day)
        else:
            max_date = datetime(2020,8,31)
        return max_date

#Enable the date picker when the camera menus have been selected
@app.callback(
    Output('date-picker', 'disabled'),
    [Input('district-dropdown', 'value'),
    Input('camera-dropdown', 'value')]
)
def search_data(district_id, camera_id):
    if district_id != None and camera_id != None:
        return False
    else:
        return True

# Show the range hour when all the fields have been selected
@app.callback(
    Output('div-slider', 'hidden'),
    [Input('date-picker', 'date'),
    Input('district-dropdown', 'value'),
    Input('camera-dropdown', 'value'),
    Input('class-dropdown', 'value')]
)
def enabled_slider(date, district_id, camera_id, class_val):
    if district_id != None and camera_id != None and class_val != None and date != None:
        return False
    else:
        return True

#Create the line chart of the camara selected and class(es)
@app.callback(
    Output('line-chart', 'children'),
    [Input('date-picker', 'date'),
    Input('hour-slider', 'value'),
    Input('district-dropdown', 'value'),
    Input('camera-dropdown', 'value'),
    Input('class-dropdown', 'value')]
)
def create_line_chart(date, hour, district_id, camera_id, class_val):
    if district_id != None and camera_id != None and class_val != None and date != None:
        date_split = date.split('-')
        date_init = datetime(int(date_split[0]), int(date_split[1]), int(date_split[2]), hour[0])
        date_fin = datetime(int(date_split[0]), int(date_split[1]), int(date_split[2]), hour[1])

        data = mongo_col.find({
            '$and': [
                {'camera_id': camera_id},
                {'district_id': district_id},
                {'timestamp': {"$gte": date_init, "$lt": date_fin}}
        ]}, sort=[('timestamp', pymongo.ASCENDING)])

        temp = {d['timestamp'].time().strftime('%H:%M:%S'): d['results'] for d in data}

        if temp != {}:
            fig = go.Figure()
            for name in class_val:
                data_chart = hour_dict(hour[0], hour[1])
                for k,v in temp.items():
                    data_chart[k] = v[name]
            
                fig.add_trace(go.Scatter(
                            x=list(data_chart.keys()),
                            y=list(data_chart.values()),
                            mode='lines+markers',
                            name=tags_rever[name],
                            line_shape='hv'))

            fig.update_layout(
                autosize=True,
                xaxis_title='Hora',
                yaxis_title='Frecuencia',
                font=dict(
                    family=font,
                    size=15
                )
            )
            div = dcc.Graph(figure=fig)
        else:
            div = html.P('No existen datos')

        return div


# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':
    app.run_server(debug=True)
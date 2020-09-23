import json
import plotly.graph_objects as go
from tqdm import tqdm
from datetime import datetime

def hour_list():
    data_chart = {}
    for i in range(15,17):
        for j in range(0,60):
            for k in range(0,60):
                date = datetime(2020,1,1,hour=i, minute=j, second=k)
                data_chart[date.time()] = None
    return data_chart

def main():
    path_names = 'data/models/YOLO/obj.names'
    #Read the class file and transform in dictionary
    with open(path_names) as f:
        names_dict = {i: line.split('\n')[0] for i,line in enumerate(f)}

    class_list = list(names_dict.values())

    with open('test/backup_hist.txt', 'r') as f:
        json_data = json.load(f)
    
    fig = go.Figure()
    for name in tqdm(class_list[:2]):
        results = []
        data_chart = hour_list()
        for k,v in json_data.items():
            data_chart[k] = v[name]
    
        fig.add_trace(go.Scatter(
                    x=list(data_chart.keys()),
                    y=list(data_chart.values()),
                    mode='lines+markers',
                    name=name,
                    line_shape='hv'))
    fig.show()

if __name__ == "__main__":
    main()
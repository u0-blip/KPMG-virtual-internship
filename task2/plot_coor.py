import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import helper
from importlib import reload
reload(helper)
from helper import *
import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from datetime import datetime, timedelta,date
from sklearn.metrics import classification_report,confusion_matrix
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import yaml

with open('param.yaml', 'rb') as f:
    params = yaml.load(f)

def plot_coor(tx_graph, f1, f2, diff_field):
    plot_data = []
    for i in range(tx_graph[diff_field].min(), tx_graph[diff_field].max()+1):
        plot_data.append(go.Scatter(
            x=tx_graph[tx_graph[diff_field]==i][f1],
            y=tx_graph[tx_graph[diff_field]==i][f2],
            mode='markers',
            name=str(i),
            marker= dict(size= 7,
                line= dict(width=1),
                opacity= 0.5
            )
        ))
    

    plot_layout = go.Layout(
            yaxis= {'title': f2},
            xaxis= {'title': f1},
            title='Segments'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.iplot(fig)

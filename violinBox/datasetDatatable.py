
import pandas as pd
import numpy as np
import dash
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash_html_components.Data import Data
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.io as pio

cir_conf_df = pd.read_csv(r'D:\ats\random_data\cirConf.csv')
nms_stat_df = pd.read_csv(r'D:\ats\random_data\nms_stat.csv')
datasetTable = pd.read_csv(r'D:\ats\random_data\dataset-vs.csv')

dataset = pd.merge(nms_stat_df, cir_conf_df, how='left', on="date")
dataset['CIR_Alloc'] = np.random.normal((np.sqrt(dataset['CIR_Conf'])*np.random.uniform(
    2, 3)), np.sqrt((dataset['CIR_Conf'])/(np.random.uniform(7, 10))))

dataset['CIR_Alloc'] = round(dataset['CIR_Alloc'], 2)
# dataset['date'] = pd.to_datetime(dataset['timestamp'])
# datasetViolin = (dataset['date'] >= "01/01/2021")


def percentil95(x):
    return np.percentile(x, 95)


ConfPrclDataset = dataset.groupby('CIR_Conf')['CIR_Alloc'].agg([
    percentil95]).reset_index()

ConfPrclDataset['percentil95'] = round(ConfPrclDataset['percentil95'], 2)
# ------------------------------------------------------------------------------
maximum_df = dataset.groupby(
    'CIR_Conf')['CIR_Alloc'].max().reset_index(name='maximum')
mean_df = dataset.groupby('CIR_Conf')[
    'CIR_Alloc'].mean().reset_index(name='mean')

prcl_df = dataset.groupby('CIR_Conf')['CIR_Alloc'].agg(
    [percentil95]).reset_index()

#--------------------------------------------------------------------------------
# Build the violin/box plot

app = dash.Dash(__name__)

app.layout = html.Div([dash_table.DataTable(
        id='dataset',
        columns=[{"name":i, "id":i, "deletable":True, "selectable":True, "hideable":True}
                 if i == "Hour" or i == "Time"
                 else{"name":i, "id":i, "deletable":False, "selectable":True}
                 for i in datasetTable[["Date", "CIR_Conf", "CIR_Alloc", "Time", "Hour"]]
                ],
    # columns=[{"name": i, "id": i} for i in datasetTable.columns],
    data=datasetTable.to_dict('records'),
    editable=False,
    filter_action="native",
    sort_action="native",
    sort_mode="single",
    column_selectable="multi",
    row_selectable="multi",
    row_deletable=True,
    selected_columns=[],
    selected_rows=[],
    page_action="native",
    page_current="0",
    # page_size="10",
    style_cell={
        'minWidth': 95, 'maxWidth': 95, 'width': 95
    },
    ),
])

if __name__ == '__main__':
    app.run_server(debug=False)

import pandas as pd
import numpy as np
import matplotlib as ptl
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.io as pio

cir_conf_df = pd.read_csv(r'D:\ats\random_data\cirConf.csv')
nms_stat_df = pd.read_csv(r'D:\ats\random_data\nms_stat.csv')

dataset = pd.merge(nms_stat_df, cir_conf_df, how='left', on="date")
dataset['CIR_Alloc'] = np.random.normal((np.sqrt(dataset['CIR_Conf'])*np.random.uniform(2, 3)), np.sqrt((dataset['CIR_Conf'])/(np.random.uniform(7, 10))))

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


line = px.line(data_frame=prcl_df, x="percentil95",y="CIR_Conf", template='plotly_white', width=1200,height=600, labels={"CIR_Conf": "CIR Configured"})
line.update_traces(mode='markers+lines')
final=line.add_trace(px.scatter(
data_frame=prcl_df, x="percentil95", y="CIR_Conf", text="CIR_Conf").data[0])
final.update_traces(textposition="bottom right")
line.update_layout(title='',
                  title_x=0.5,
                  showlegend=False)
line.update_layout(
    yaxis=dict(tickfont=dict(size=12)),
    xaxis=dict(tickfont=dict(size=12)),    
    font=dict(family="Courier New, monospace",size=12,color="black"
    ))
pio.show(line)

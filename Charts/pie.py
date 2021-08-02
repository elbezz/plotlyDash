import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import plotly.io as pio

df=pd.read_csv("Chart/covid-19-states-daily.csv")
df= df[df['dateChecked']== '2020-03-17']
df=df[df['death']>=5]
print(df)

pie_chart=px.pie(
    data_frame=df,
    values='death',
    names='state',
    color='state',
    color_discrete_sequence=['red','green','blue','orange'],
    #color_discrete_map={"WA":"yellow","CA":"red","NY":"black","FL":"brown"},
    hover_name='negative', # value appears on top in bold when you hover
    hover_data=['positive'], #extra data in hover
    
    # custom_data=['total'] 
    # this is gonna be memorized to be used in callbacks funtions

    labels={'state':'The State'},
    title='Coronavirus in the USA',
    template='gridon',
           #'ggplot2', 'seaborn', 'simple_white', 'plotly',
        #'plotly_white', 'plotly_dark', 'presentation',
        #'xgridoff', 'ygridoff', 'gridon', 'none'


    width=800,                         
    height=600,                         
    hole=0.5, 
)

pie_chart.update_traces(textposition='outside', textinfo='percent+label',
                        # marker=dict(line=dict(color='#000000', width=4)),
                        # pull=[0, 0, 0.2, 0], opacity=0.7, rotation=180
                        )
pio.show(pie_chart)
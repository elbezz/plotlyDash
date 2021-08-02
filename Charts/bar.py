import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import pandas_datareader.data as web
import datetime
import plotly.io as pio

df = pd.read_csv('Chart/Caste.csv')
df = df[df['state_name']=='Maharashtra']
df = df.groupby(['year','gender',], as_index=False)[['detenues','under_trial','convicts','others']].sum()
print(df[:5])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )


barchart = px.bar(
    data_frame=df,
    x='year',
    # x='gender', 
    y='convicts',
    color='gender',
    # change it later to 'convicts'
    # and add these two lines under 
    # color_discrete_map={"Male": "gray","Female": "pink"}(comment it out)
    # the two lines are:
    #   color_continuous_scale=px.colors.diverging.Picnic,
    #   range_color=[1,10000],
    opacity=0.9,
    orientation='v',
    # 'h' to horizontal... you must swap x and y too
    barmode='relative',
    # try also 'overlay' 'group'
    
    ########################################################## 
    # because we don't have the 'cast' column in the data df
    # comment out the the groupby df and try facet_row='cast'

    # facet_row='caste',

    # facet_col='caste',
    # facet_col_wrap=2,

    #This will define the maximum of sublot columns... dont use facet_row with these above two!
    ##########################################################
# color_discrete_sequence=('yellow','pink')
# color_discrete_map={"Male": "gray","Female": "pink"},

# befor using this section change color='gender', to color='convicts',
    # color_continuous_scale=px.colors.diverging.Picnic,
    # range_color=[1,10000],

    # check more on   https://plotly.com/python/builtin-colorscales/

#################################################################################
# uncomment and use..

text='convicts',
hover_name='under_trial',
hover_data=['detenues'],
custom_data=['others'],

# custom_data=['others'],   This is invisible data and it's gonna be used in @callbacks
##################################################################################

labels={"convicts":"Convicts in Maharashtra","gender":"Gender", "year":"Year"},
title="Indian prison stats",
width=1200,
height=720,
template="seaborn",
#  check also template =
# 'ggplot2', 'seaborn', 'simple_white', 'plotly',
# 'plotly_white', 'plotly_dark', 'presentation',
# 'xgridoff', 'ygridoff', 'gridon', 'none'

###############################################################################################
###############################################################################################
# ANIMATION
# befor using this section, go on the top and change the x='gender', because the play button
# is already going to be year

##############################################################################################
# animation_frame='year',
# range_y=[0,9000],
# category_orders={'year':[2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001]}
##############################################################################################

# we didn't use range_x=[,], becauae the x axis is male and female, so dont forget
# to set up if needed

###############################################################################################
###############################################################################################

# plotly.graph_objects.Layout CHECK MORE ON
# https://plotly.com/python-api-reference/generated/plotly.graph_objects.Layout.html#plotly.graph_objects.Layout


)

# use these two bellow to control the animation timing
#####################################################################################
# barchart.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000
# barchart.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500
#####################################################################################


barchart.update_layout(uniformtext_minsize=14, uniformtext_mode='hide',
                       legend={'x':0,'y':1.0}),

###########################################################################
# Try this too!!!!!!!!

# barchart.update_traces(texttemplate='%{text:.2s}', textposition='outside',
#                        width=[.3,.3,.3,.3,.3,.3,.6,.3,.3,.3,.3,.3,.3])
pio.show(barchart)
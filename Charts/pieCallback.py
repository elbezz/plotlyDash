from dash_html_components.Div import Div
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


df = pd.read_csv('Charts/Urban_Park_Ranger_Animal_Condition_Response.csv')

# you need to include __name__ in your Dash constructor if
# you plan to use a custom CSS or JavaScript in your Dash apps
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],)

app.layout = html.Div([
    html.Div([
        html.Label(['NYC Calls for animal rescue']),
        dcc.Dropdown(
            id='mydropdown',
            options=[
                {'label': 'Action Taken by Ranger',
                    'value': 'Final Ranger Action'},
                {'label': 'Age', 'value': 'Age'},
                {'label': 'Animal Health', 'value': 'Animal Condition'},
                {'label': 'Borough', 'value': 'Borough'},
                {'label': 'Species', 'value': 'Animal Class'},
                {'label': 'Species Status', 'value': 'Species Status'}
            ],
            value='Animal Class',
            multi=False,
            clearable=False,
            style={"width": "50%"}
        ),
    ]),

        html.Div([
        dcc.Graph(id='the_graph')
    ]),
])

@app.callback(
    Output(component_id='the_graph', component_property='figure'),
    [Input(component_id='mydropdown', component_property='value')]
)

def update_graph(mydropdown):
    dff=df
    piechart=px.pie(
        data_frame=dff,
        names=mydropdown,
        hole=.3,
    )
    return (piechart)

if __name__ == '__main__':
    app.run_server(debug=False)
import pandas as pd
import numpy as np
import dash
import os
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash_html_components.Data import Data
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import plotly
import plotly.express as px
import plotly.io as pio
import dash_table as dt
from scipy.sparse.linalg import lsqr as sparse_lsqr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
cir_conf_df = pd.read_csv(r'D:\ats\random_data\cirConf.csv')

# cir_conf_time_df = cir_conf_df.copy()
# cir_conf_time_df['date'] = pd.to_datetime(
# cir_conf_time_df['date'], dayfirst=True)
# cir_conf_time_df = cir_conf_time_df.set_index('date')
# cir_conf_time_df = cir_conf_time_df.asfreq('M')
# print(cir_conf_time_df)

nms_stat_df = pd.read_csv(r'D:\ats\random_data\nms_stat.csv')
datasetTable = pd.read_csv(r'D:\ats\random_data\dataset-vs.csv')

dataset = pd.merge(nms_stat_df, cir_conf_df, how='left', on="date")
dataset['CIR_Alloc'] = np.random.normal((np.sqrt(dataset['CIR_Conf'])*np.random.uniform(
    2, 3)), np.sqrt((dataset['CIR_Conf'])/(np.random.uniform(7, 10))))
dataset['CIR_Alloc'] = round(dataset['CIR_Alloc'], 2)

prcl_df = pd.DataFrame(columns=["CIR-Configured", "percentile"])


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6('ATS Overbooking system'),
                ], style={'textAlign': 'right', 'margin-top': '0px'}
                )
            ], style={"border-radius": "2%", "background": "light", 'height': '120px', 'border': 'none','margin-top': '20px'}),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6('Select Pool'),
                    dcc.Dropdown(id="slct_pool",
                                 options=[
                                     {"label": "Azal", "value": "Azal"},
                                     {"label": "Pool1", "value": "Pool1"},
                                     {"label": "Pool2", "value": "Pool2"}],
                                 multi=False,
                                 value="Azal"),
                ])], style={"border-radius": "2%", "background": "primary", 'outline': 'True', 'textAlign': 'left', 'margin-top': '20px', 'height': '120px'}),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6('Select Percentil %'),
                    dcc.Dropdown(id="slct_percentil",
                             options=[
                                 {"label": "90", "value": 90},
                                 {"label": "91", "value": 91},
                                 {"label": "92", "value": 92},
                                 {"label": "93", "value": 93},
                                 {"label": "94", "value": 94},
                                 {"label": "95", "value": 95},
                                 {"label": "96", "value": 96},
                                 {"label": "97", "value": 97},
                                 {"label": "98", "value": 98},
                                 {"label": "99", "value": 99}],
                             multi=False,
                             value=""),
                ])], style={"border-radius": "2%", "background": "primary", 'outline': 'True', 'textAlign': 'left', 'margin-top': '20px', 'height': '120px'}),
            ], width=2),
        dbc.Col([
            dbc.Card([
                 dbc.CardBody([
                    html.H6('Select CIR_Physic'),
                        dcc.Dropdown(id="slct_CIR_Phy",
                             options=[
                                 {"label": "50", "value": 50},
                                 {"label": "60", "value": 60},
                                 {"label": "70", "value": 70},
                                 {"label": "80", "value": 80},
                                 {"label": "90", "value": 90},
                                 {"label": "100", "value": 100},
                                 {"label": "110", "value": 110},
                                 {"label": "120", "value": 120},
                                 {"label": "130", "value": 130},
                                 {"label": "140", "value": 140},
                                 {"label": "150", "value": 150},
                                 {"label": "160", "value": 160},
                                 {"label": "170", "value": 170},
                                 {"label": "180", "value": 180},
                                 {"label": "190", "value": 190},
                                 {"label": "200", "value": 200},
                                 {"label": "210", "value": 210},
                                 {"label": "220", "value": 220},
                                 {"label": "230", "value": 230},
                                 {"label": "240", "value": 240},
                                 {"label": "250", "value": 250},
                                 {"label": "260", "value": 260},
                                 {"label": "270", "value": 270},
                                 {"label": "280", "value": 280},
                                 {"label": "290", "value": 290},
                                 {"label": "300", "value": 300}],
                             multi=False,
                             value=50),
                     ])], style={"border-radius": "2%", "background": "primary", 'outline': 'True', 'textAlign': 'left', 'margin-top': '20px', 'height': '120px'}),
    ], width=2),
        dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Percentil'),
                        html.Br(),
                        html.H2(id='content-Percentil', children="-")
                    ], style={'textAlign': 'center'})

                ], style={"border-radius": "8%", "background": "PowderBlue", 'textAlign': 'center', 'margin-top': '20px', 'height': '120px'}),
            ], width=2),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Current CIR'),
                        html.Br(),
                        html.H2(id='content-CIR-Conf', children="-")
                    ], style={"border-radius": "8%", "background": "PowderBlue", 'textAlign': 'center', 'margin-top': '20px', 'height': '120px'})
                ], color="light", outline=True),
            ], width=2),
            ], className='mb-2 mt-2'),

    dbc.Row([
        dbc.Col([            
                dcc.Graph(id='violin-chart-cir-conf-cir-alloc-percentil', figure={}), 
            ], width=12),
            ]),

    dbc.Row([
        dbc.Col([            
                dcc.Graph(id='line-chart-cir-conf-percentil', figure={}),    
            ], width=10),

        dbc.Col([
                html.Br(),
                html.Br(),
                html.Br(),   
                    dt.DataTable(
                        id='tblCurrent', data=prcl_df.to_dict('records'),columns=[{"name": i, "id": i}
                        for i in prcl_df.columns],
                        style_table={'height': '600px'},
                        style_cell={'minWidth': 95, 'maxWidth': 95, 'width': 95},
                    )
        ], width=2)
    ]),


    dbc.Row([
        
        dbc.Col([             
                    dcc.Graph(id='line-chart-cir-conf-percentil-prediction', figure={}),
            ], width=10),
        
        dbc.Col([     
            html.Br(),
            html.Br(),
            dbc.Card([
                dbc.CardBody([
                    html.H5(id='content-regression-equation', children="")
                ], style={"border-radius": "0%", "background": 'PowderBlue', 'inverse': 'True', 'textAlign': 'left', 'margin-top': '0px', 'height': '60px'})
            ], color="light", outline=True),
            dt.DataTable(
                id='tblPred', data=prcl_df.to_dict('records'), columns=[{"name": i, "id": i}
                for i in prcl_df.columns],
                style_table={'height': '600px'},style_cell={'minWidth': 95, 'maxWidth': 95, 'width': 95},)
            ], width=2),
            ]),

    dbc.Row([
        dbc.Col([
           
        ], width=2),

        dbc.Col([
                dcc.Graph(id='pie-chart-cir-conf-prediction', figure={}),
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6('Remaining Time'),
                    html.H3(id='content-RemainingTime', children="-")
                ], style={"border-radius": "8%", "background": 'rgba(50, 250, 120, 1.0)', 'inverse': 'True', 'textAlign': 'center', 'margin-top': '0px', 'height': '120px'})
            ], color="light", outline=True),
        ], width=2),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6('Overbooking Ratio'),
                    html.H3(id='content-OverbookingRatio', children="-")
                ], style={"border-radius": "8%", "background": 'rgba(50, 250, 120, 1.0)', 'inverse': 'True', 'textAlign': 'center', 'margin-top': '0px', 'height': '120px'})
            ], color="light", outline=True),
        ], width=2),
   
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='line-sales_forecast', figure={}),
            ], width=12),
            ]),

], fluid=True)

# Bar Chart1 ************************************************************
#Updating top cards ******************************************
@app.callback(
    Output('content-Percentil', 'children'),
    Output('content-CIR-Conf', 'children'),
    Output('content-RemainingTime', 'children'),
    Output('content-OverbookingRatio', 'children'),
    Output('content-regression-equation', 'children'),
    Output('violin-chart-cir-conf-cir-alloc-percentil','figure'),
    Output('line-chart-cir-conf-percentil', 'figure'),
    Output('line-chart-cir-conf-percentil-prediction', 'figure'), 
    Output('pie-chart-cir-conf-prediction', 'figure'),
    Output('tblPred', 'data'),
    Output('tblCurrent', 'data'), 
    Output('line-sales_forecast', 'figure'),
    Input('slct_percentil', 'value'),
    Input('slct_CIR_Phy', 'value') 
)
def update_small_cards(option_slctd, option_slctd2):
    percentil = option_slctd
    def percentile(x):
        return np.percentile(x, percentil)
    datasetCopy=dataset.copy()
    dff = cir_conf_df.copy()
    max_cr_conf = dff.CIR_Conf.max()

    dffLine = datasetCopy.groupby('CIR_Conf')['CIR_Alloc'].agg(
    [percentile]).reset_index()
    dffLine['percentile'] = round(dffLine['percentile'], 2)
    dffLine.rename(columns={"CIR_Conf": "CIR-Configured"}, inplace=True)

# the liniar regression
    linear_model_sqrt = LinearRegression(normalize=True)
    x = dffLine[["CIR-Configured"]]
    y = dffLine[["percentile"]]
    X_sqrt_train_transform = np.sqrt(x)
    linear_model_sqrt.fit(X_sqrt_train_transform, y)
    y_pred_train_sqrt = linear_model_sqrt.predict(X_sqrt_train_transform)

    linear_model_square_inv = LinearRegression(normalize=True)
    x_inv = dffLine[["percentile"]]
    y_inv = dffLine[["CIR-Configured"]]
    X_square_train_transform_inv = x_inv*x_inv
    linear_model_square_inv.fit(X_square_train_transform_inv, y_inv)
    y_pred_train_square_inv = linear_model_square_inv.predict(
        X_square_train_transform_inv)

# ------------------------------------------------------------------------------
# join regression_df for the plot
    y_pred_train_sqrt_df = pd.DataFrame(
        y_pred_train_sqrt, columns=['y_pred_train_sqrt'])
    y_pred_train_sqrt_df['index'] = y_pred_train_sqrt_df.index

    y_pred_train_sqrt_inv_df = pd.DataFrame(
        y_pred_train_square_inv, columns=['y_pred_train_sqrt_inv'])
    y_pred_train_sqrt_inv_df['index'] = y_pred_train_sqrt_inv_df.index

    x['index'] = x.index
    y['index'] = y.index

    regression_df = pd.merge(pd.merge(pd.merge(
        x, y, on='index'), y_pred_train_sqrt_df, on='index'), y_pred_train_sqrt_inv_df, on='index')

    regression_df.rename(
        columns={"y_pred_train_sqrt_inv": "CIR-Configuredd"}, inplace=True)
    regression_df['CIR-Configuredd'] = round(
        regression_df['CIR-Configuredd'], 2)

    predictionTbl_df = regression_df[["CIR-Configuredd", "percentile"]].copy()
    predictionTbl_df.rename(
        columns={"CIR-Configuredd": "CIR-Configured"}, inplace=True)
# ---------------------------------------------------------------------------------------
# Prédiction CIR_Conf Max
    CIR_Phy = [[option_slctd2]]
# Prédire maintenant CIR_Conf quand percentile = CIR_Phy avec le modèle inversé qu'on vient de mettre en place
    y_pred_CIR_Conf_Max = linear_model_square_inv.predict(np.square(CIR_Phy))
    y_pred_CIR_Conf_Max_round = round(int(y_pred_CIR_Conf_Max[0][0]),2)

# Pie dataframe
    column_names = ["type", "value"]
    pie_df = pd.DataFrame(columns=column_names)
    pie_df = pie_df.append(
    {'type': 'max_cr_conf', 'value': max_cr_conf}, ignore_index=True)
    pie_df = pie_df.append({'type': 'max_cr_conf_phy', 'value': int(
    y_pred_CIR_Conf_Max[0][0])-max_cr_conf}, ignore_index=True)
    left = int(y_pred_CIR_Conf_Max[0][0])-max_cr_conf


# Prédiction du temps qui reste pour atteindre CIR_Conf Max depuis le modèle Time Série "évolution des ventes en fonction du temps"
    cir_conf_time_df = dff.copy()
    cir_conf_time_df['date'] = pd.to_datetime(
    cir_conf_time_df['date'], dayfirst=True)
    cir_conf_time_df = cir_conf_time_df.set_index('date')
    cir_conf_time_df = cir_conf_time_df.asfreq('M')
   
    model_time = HWES(cir_conf_time_df, seasonal_periods=6,
                  trend='add', seasonal='add', freq='M')
    fitted = model_time.fit(optimized=True, use_brute=True)
    sales_forecast = fitted.forecast(steps=18)

    sales_forecast_df = pd.DataFrame(sales_forecast, columns=['predicted'])
    sales_forecast_df.index.names = ["date"]
    sales_forecast_df['predicted'] = round(sales_forecast_df['predicted'], 2)

    selection_df = sales_forecast_df[(
    sales_forecast_df.predicted >= y_pred_CIR_Conf_Max_round)]
    Timelimite = selection_df.index.min()
    RT= pd.Timedelta(((Timelimite-cir_conf_time_df.index[-1])), unit='D').days
    RT= (str(RT) + " Days")
    # print(RT)
    Overbooking_Ratio_array = y_pred_CIR_Conf_Max[0][0]/CIR_Phy
    Overbooking_Ratio = round(Overbooking_Ratio_array[0][0], 2)

    a = linear_model_square_inv.coef_[0][0]
    b = linear_model_square_inv.intercept_[0]

    # newline = '\n'
    # equation = f"y = ax² + b{newline}    y ={str(round(a, 2)) } x² +{str(round(b, 2))}"
    equation = "y =" + str(round(a, 2)) + "x² +"+str(round(b, 2))

#-----------------------------------------------------------------------------------------
    trace1 = px.violin(data_frame=datasetCopy, x="CIR_Conf", y="CIR_Alloc", orientation="v", box=True, color='CIR_Conf', log_x=True, log_y=True, labels={"CIR_Conf": "CIR Configured", "CIR_Alloc": "CIR Allocated"}, width=1500, height=600, template='plotly_dark', animation_frame='CIR_Conf', range_x=[10, 300], range_y=[5, 70],).update_layout(yaxis=dict(tickfont=dict(size=1)), xaxis=dict(tickfont=dict(size=1)), font=dict(
        family="Courier New, monospace", size=10, color="white")).add_trace(px.line(dffLine, x="CIR-Configured", y="percentile").update_traces(mode='markers+lines')
  .update_layout(title='', title_x=0.5, showlegend=False,).update_layout(yaxis=dict(tickfont=dict(size=12)), xaxis=dict(tickfont=dict(size=12)), font=dict(family="Courier New, monospace", size=10, color="black")).data[0]) 
    trace2 = px.line(data_frame=dffLine, x="percentile", y="CIR-Configured", template='presentation', width=1300, height=600, labels={"CIR_Conf": "CIR Configured", "percentile": "percentile-" + str(percentil)}).update_traces(
        mode='markers+lines').add_trace(px.scatter(data_frame=dffLine, x="percentile", y="CIR-Configured", text="CIR-Configured").data[0]).update_traces(textposition="bottom right").update_layout(title='', title_x=0.5, showlegend=False,).update_layout(yaxis=dict(tickfont=dict(size=12)),xaxis=dict(tickfont=dict(size=12)),font=dict(family="Courier New, monospace",size=10,color="black"))
    trace3 = px.line(data_frame=predictionTbl_df, x="percentile", y="CIR-Configured", template='ggplot2', width=1300, height=600).update_traces(mode='markers+lines').add_trace(px.scatter(data_frame=predictionTbl_df, x="percentile", y="CIR-Configured", text="CIR-Configured")
    .data[0]).update_traces(textposition="bottom right").update_layout(title='', title_x=0.5, showlegend=False,).update_layout(yaxis=dict(tickfont=dict(size=12)), xaxis=dict(tickfont=dict(size=12)), font=dict(family="Courier New, monospace", size=10, color="black"))

    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
    trace4 = px.pie(pie_df, values=[max_cr_conf, left], names=['Current CIR-conf', 'Left'], template='presentation', title='').update_traces(
        hoverinfo='label+percent', textinfo='percent+value', textfont_size=20, marker=dict(colors=colors, line=dict(color='#000000', width=2)), pull=(0, 0.1))

    trace5 = px.line(data_frame=cir_conf_time_df, x=cir_conf_time_df.index, y="CIR_Conf", template='plotly_dark', width=1500, height=600, labels={"CIR_Conf": "CIR Configured"}).update_traces(mode='markers+lines').add_trace(px.line(data_frame=sales_forecast_df, x=sales_forecast_df.index, y="predicted", text="predicted").data[0]).update_traces(textposition="bottom right").update_layout(title='', title_x=0.5, showlegend=False,).update_layout(
            yaxis=dict(tickfont=dict(size=12)),
            xaxis=dict(tickfont=dict(size=12)),
            font=dict(family="Courier New, monospace", size=10, color="yellow"))

    return percentil, max_cr_conf, RT, Overbooking_Ratio, equation, trace1, trace2, trace3, trace4, predictionTbl_df[["CIR-Configured", "percentile"]].to_dict('records'), dffLine.to_dict('records'), trace5


if __name__ == '__main__':
    app.run_server(debug=False, port=8000)

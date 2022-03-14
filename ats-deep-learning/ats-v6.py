import pandas as pd
import numpy as np
import dash
import os
from dash.dependencies import Input, Output
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
from sklearn.preprocessing import PolynomialFeatures
cir_conf_df = pd.read_csv(r'E:\ats\ats\random_data\cirConf.csv')

nms_stat_df = pd.read_csv(r'E:\ats\ats\random_data\nms_stat.csv')
datasetTable = pd.read_csv(r'E:\ats\ats\random_data\dataset-vs.csv')

dataset = pd.merge(nms_stat_df, cir_conf_df, how='left', on="date")
dataset['CIR_Alloc'] = np.random.normal((np.sqrt(dataset['CIR_Conf'])*np.random.uniform(
    2, 3)), np.sqrt((dataset['CIR_Conf'])/(np.random.uniform(7, 10))))
dataset['CIR_Alloc'] = round(dataset['CIR_Alloc'], 2)
prcl_df = pd.DataFrame(columns=["Bandwidth Sold", "Bandwidth Required"])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H1('Overbooking Manager'),
                ], style={'textAlign': 'center', 'margin': '0px'}
                )
            ], style={"border-radius": "0%", "background": "#023047", 'height': '80px', 'border': 'none', 'margin-top': '0px', 'color': 'white'}),
        ], width=12),
    ], className='mb-2 mt-2'),


    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6('Select Bandwidth Pool '),
                    dcc.Dropdown(id="slct_pool",
                                 options=[
                                     {"label": "Azal", "value": "Azal"},
                                     {"label": "Pool1", "value": "Pool1"},
                                     {"label": "Pool2", "value": "Pool2"}],
                                 multi=False,
                                 value="Azal"),
                ])], style={"border-radius": "2%", "background": "primary", 'outline': 'True', 'textAlign': 'left', 'margin': '0px', 'height': '100px'}),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6('Bandwidth Availability(%)'),
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
                ])], style={"border-radius": "2%", "background": "primary", 'outline': 'True', 'textAlign': 'left', 'margin': '0px', 'height': '100px'}),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                     html.H6('Physical Bandwidth Available'),
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
                     ])], style={"border-radius": "2%", "background": "primary", 'outline': 'True', 'textAlign': 'left', 'margin': '0px', 'height': '100px'}),
        ], width=4),
    ], className='mb-2 mt-2'),
    dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Bandwidth Sold'),
                        html.H2(id='content-CIR-Conf', children="-")
                    ], style={"border-radius": "3%", "background": "#8ecae6", 'textAlign': 'center', 'margin': '0px', 'height': '120px', 'color': '#023047'})
                ], color="light", outline=True),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Max Bandwidth to be sold'),
                        html.H2(id='max-bandwidth-to-be-sold', children="-")
                    ], style={'textAlign': 'center'})

                ], style={"border-radius": "3%", "background": "#8ecae6", 'textAlign': 'center', 'margin-top': '0px', 'height': '120px', 'color': '#023047'}),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Remaining Bandwidth'),
                        html.H2(id='Remaining-Bandwidth', children="-")
                    ], style={'textAlign': 'center'})

                ], style={"border-radius": "3%", "background": "#8ecae6", 'textAlign': 'center', 'margin-top': '0px', 'height': '120px', 'color': '#023047'}),
            ], width=4),
            ], className='mb-2 mt-2'),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Remaining Time'),
                    html.H3(id='content-RemainingTime', children="-")
                ], style={"border-radius": "3%", "background": '#219ebc', 'inverse': 'True', 'textAlign': 'center', 'margin': '0px', 'height': '120px', 'color': '#023047'})
            ], color="light", outline=True),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Planning for new BW acquisition'),
                    html.H3(id='planning-for-new-BW-acquisition', children="-")
                ], style={"border-radius": "3%", "background": '#219ebc', 'inverse': 'True', 'textAlign': 'center', 'margin': '0px', 'height': '120px', 'color': '#023047'})
            ], color="light", outline=True),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Overbooking Ratio'),
                    html.H3(id='content-OverbookingRatio', children="-")
                ], style={"border-radius": "3%", "background": '#219ebc', 'inverse': 'True', 'textAlign': 'center', 'margin': '0px', 'height': '120px', 'color': '#023047'})
            ], color="light", outline=True),
        ], width=4),
    ], className='mb-2 mt-2'),

    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='line-chart-cir-conf-alloc-perc-date', figure={}),
        ], width=6),
        dbc.Col([
                dcc.Graph(
                    id='violin-chart-cir-conf-cir-alloc-percentil', figure={}),
                ], width=6),
    ], className='mb-2 mt-2'),
    dbc.Row([
            dbc.Col([
                dcc.Graph(id='line-sales_forecast', figure={}),
            ], width=12),
            ], className='mb-2 mt-2'),

    dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='line-chart-cir-conf-percentil-prediction', figure={}),
            ], width=8),
            dbc.Col([
  
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(id='content-regression-equation',
                                    children="")
                        ], style={"border-radius": "0%", "background": '#8ecae6', 'inverse': 'True', 'textAlign': 'left', 'margin': '0px', 'height': '60px'})
                    ], color="light", outline=True),
                    dt.DataTable(
                        id='tblPred', data=prcl_df.to_dict('records'), columns=[{"name": i, "id": i}for i in prcl_df.columns],
                        style_table={'height': '600px'}, style_cell={'minWidth': 95, 'maxWidth': 95, 'width': 95},)
                    ], width=4),
            ], className='mb-2 mt-2'),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='pie-chart-cir-conf-prediction', figure={}),
        ], width=12),
    ], className='mb-2 mt-2'),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='line-chart-overbooking-space', figure={}),
        ], width=12),
    ], className='mb-2 mt-2'),



], fluid=True)
# #######################################################################################
@app.callback(
    Output('content-CIR-Conf', 'children'),
    Output('content-RemainingTime', 'children'),
    Output('content-OverbookingRatio', 'children'),
    Output('content-regression-equation', 'children'),
    Output('max-bandwidth-to-be-sold', 'children'),
    Output('Remaining-Bandwidth', 'children'),
    Output('violin-chart-cir-conf-cir-alloc-percentil', 'figure'),
    Output('line-chart-cir-conf-percentil-prediction', 'figure'),
    Output('pie-chart-cir-conf-prediction', 'figure'),
    Output('tblPred', 'data'),
    Output('planning-for-new-BW-acquisition', 'children'),
    Output('line-sales_forecast', 'figure'),
    Output('line-chart-cir-conf-alloc-perc-date', 'figure'),
    Output('line-chart-overbooking-space', 'figure'),

    Input('slct_percentil', 'value'),
    Input('slct_CIR_Phy', 'value')
)
def update_small_cards(option_slctd, option_slctd2):
    percentil = option_slctd

    def percentile(x):
        return np.percentile(x, percentil)
    datasetCopy = dataset.copy()
    dff = cir_conf_df.copy()
    max_cr_conf = dff.CIR_Conf.max()

    dffLine = datasetCopy.groupby('CIR_Conf')['CIR_Alloc'].agg(
        [percentile]).reset_index()
    dffLine['percentile'] = round(dffLine['percentile'], 1)
    dffLine['diff'] = round((dffLine['CIR_Conf'] - dffLine['percentile']) /
                            dffLine['CIR_Conf']*100, 1).astype(str) + '%'
   

    # print(dffLine)

    conf_alloc_perc_df = pd.merge(
        datasetCopy, dffLine, how='left', on="CIR_Conf")
    new1=conf_alloc_perc_df.groupby(['date', 'CIR_Conf']).max('CIR_Alloc').reset_index()
   
    new2=conf_alloc_perc_df.copy()
    conf_alloc_perc_df.rename(
        columns={"CIR_Conf": "Sold", "date": "Time", "CIR_Alloc": "Requested", "percentile": "Required" + "(" + str(percentil)+"%)"}, inplace=True)

    conf_alloc_perc_df =conf_alloc_perc_df.loc[0:len(
        conf_alloc_perc_df):10000]
    # print(conf_alloc_perc_df)



    new1["i"] = new1["date"] + new1["CIR_Conf"].astype(str)
    new1.drop(['date','CIR_Conf','percentile'], axis=1,inplace=True)

    
    new2["i"] = new2["date"] + new2["CIR_Conf"].astype(str)
    new2.drop(['CIR_Alloc'], axis=1,inplace=True)

    plot_dff = pd.merge(new2, new1, how='left', on="i")
    plot_dff.drop(['i'], axis=1,inplace=True)

    plot_dff.rename(
        columns={"CIR_Conf": "Sold", "date": "Time", "CIR_Alloc": "Requested", "percentile": "Required" + "(" + str(percentil)+"%)"}, inplace=True)
    plot_dff =plot_dff.loc[0:len(
        plot_dff):10000]
    print('plotDF')
    print(plot_dff)


    dffLine.rename(columns={"CIR_Conf": "CIR-Configured"}, inplace=True)
# The linear regression
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
# ///////////////////////////////////////////////////////////////////////
#///////////////////////////////////////////////////////////////////////
    linear_model_poly = LinearRegression(normalize=True)
    poly_reg=PolynomialFeatures(degree=2)
    X_poly=poly_reg.fit_transform(x_inv)
    poly_reg.fit(X_poly,y_inv)
    linear_model_poly.fit(X_poly,y_inv) 
    y_pred_train_poly = linear_model_poly.predict(X_poly)

    linear_model_poly_inv = LinearRegression(normalize=True)
    # x_inv = dffLine[["percentile"]]
    # y_inv = dffLine[["CIR-Configured"]]
    X_poly_train_transform_inv = x_inv*x_inv
    linear_model_poly_inv.fit(X_poly_train_transform_inv, y_inv)
    y_pred_train_poly_inv = linear_model_square_inv.predict(
        X_poly_train_transform_inv)
    # print ("Square regression MSE Training : "+str(mean_squared_error(y_inv, y_pred_train_poly)))
    # c = linear_model_poly.coef_[0][2]
    # d = linear_model_poly.coef_[0][1]
    # e = linear_model_poly.intercept_[0]
    # print ("y = ax² + bx + c")
    # print ("y = " + str(round(c,4)) + "x²" + " + " + str(round(d,4)) + "x" + " + " + str(round(e,4)))

    y_pred_train_poly_df = pd.DataFrame(
        y_pred_train_poly, columns=['y_pred_train_poly'])

    y_pred_train_poly_inv_df = pd.DataFrame(
        y_pred_train_poly_inv, columns=['y_pred_train_poly_inv'])

    regression_poly_df = pd.merge(pd.merge(pd.merge(
        x, y, left_index=True, right_index=True), y_pred_train_poly_df, left_index=True, right_index=True),y_pred_train_poly_inv_df, left_index=True, right_index=True)

    regression_poly_df.rename(
        columns={"y_pred_train_poly_inv": "CIR-Configuredd"}, inplace=True)
    regression_poly_df['CIR-Configuredd'] = round(
        regression_poly_df['CIR-Configuredd'], 2)

    predictionTbl_poly_df = regression_poly_df[["CIR-Configuredd", "percentile"]].copy()
    predictionTbl_poly_df.rename(
        columns={"CIR-Configuredd": "Bandwidth Sold", "percentile": "Bandwidth Required"}, inplace=True)
    print(predictionTbl_poly_df)
    
    CIR_Phy = [[option_slctd2]]
    y_pred_CIR_Conf_Max = linear_model_poly_inv.predict(np.square(CIR_Phy))
    y_pred_CIR_Conf_Max_round = round(int(y_pred_CIR_Conf_Max[0][0]), 2)



    #///////////////////////////////////////////////////////////////////////////////////////////////////////
# ------------------------------------------------------------------------------
# join regression_df for the plot
    y_pred_train_sqrt_df = pd.DataFrame(
        y_pred_train_sqrt, columns=['y_pred_train_sqrt'])
    # y_pred_train_sqrt_df['index'] = y_pred_train_sqrt_df.index

    y_pred_train_sqrt_inv_df = pd.DataFrame(
        y_pred_train_square_inv, columns=['y_pred_train_sqrt_inv'])
    # y_pred_train_sqrt_inv_df['index'] = y_pred_train_sqrt_inv_df.index

    # x['index'] = x.index
    # y['index'] = y.index

    regression_df = pd.merge(pd.merge(pd.merge(
        x, y, left_index=True, right_index=True), y_pred_train_sqrt_df, left_index=True, right_index=True), y_pred_train_sqrt_inv_df, left_index=True, right_index=True)

    regression_df.rename(
        columns={"y_pred_train_sqrt_inv": "CIR-Configuredd"}, inplace=True)
    regression_df['CIR-Configuredd'] = round(
        regression_df['CIR-Configuredd'], 2)

    predictionTbl_df = regression_df[["CIR-Configuredd", "percentile"]].copy()
    predictionTbl_df.rename(
        columns={"CIR-Configuredd": "Bandwidth Sold", "percentile": "Bandwidth Required"}, inplace=True)

    # print(regression_df)
# ---------------------------------------------------------------------------------------
# Prédiction CIR_Conf Max
    # CIR_Phy = [[option_slctd2]]
# Prédire maintenant CIR_Conf quand percentile = CIR_Phy avec le modèle inversé qu'on vient de mettre en place
    # y_pred_CIR_Conf_Max = linear_model_square_inv.predict(np.square(CIR_Phy))
    # y_pred_CIR_Conf_Max_round = round(int(y_pred_CIR_Conf_Max[0][0]), 2)

# Pie chart's dataframe
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
    # print(sales_forecast_df)

    planning_BW_acquisition = sales_forecast_df.loc[sales_forecast_df['predicted']
                                                    <= y_pred_CIR_Conf_Max_round]

    planning_BW_acquisition_up_to_max = sales_forecast_df.loc[
        sales_forecast_df['predicted'] >= y_pred_CIR_Conf_Max_round]
    if planning_BW_acquisition.empty:
        planning_BW_acquisition_value = 0
    else:
        planning_BW_acquisition_value = planning_BW_acquisition.index[-1].date(
        )
    frames = [planning_BW_acquisition.tail(1),
              planning_BW_acquisition_up_to_max]
    planning_BW_acquisition_up_to_max = pd.concat(frames)
    # print(sales_forecast_df)
    # print(planning_BW_acquisition)
    # print(type(planning_BW_acquisition_value))
    # print(planning_BW_acquisition_value)
    # print(planning_BW_acquisition_up_to_max)
    selection_df = sales_forecast_df[(
        sales_forecast_df.predicted >= y_pred_CIR_Conf_Max_round)]
    Timelimite = selection_df.index.min()
    RT = pd.Timedelta(((Timelimite-cir_conf_time_df.index[-1])), unit='D').days
    RT = (str(RT) + " Days")
    # print(RT)
    Overbooking_Ratio_array = y_pred_CIR_Conf_Max[0][0]/CIR_Phy
    Overbooking_Ratio = round(Overbooking_Ratio_array[0][0], 2)

    # a = linear_model_poly_inv.coef_[0][0]
    # b = linear_model_poly_inv.intercept_[0]
    c = linear_model_poly.coef_[0][2]
    d = linear_model_poly.coef_[0][1]
    e = linear_model_poly.intercept_[0]

    equation = "y = " + str(round(c,4)) + "x²" + " + " + str(round(d,4)) + "x" + " + " + str(round(e,4))
    # equation = "y =" + str(round(a, 2)) + "x² +"+str(round(b, 2))
#//////////////////////////////////////////////////////////////////////////
    columns_names = ["Bandwidth Required", "Bandwidth Sold"]
    phy_maxBW_poly_df = pd.DataFrame(columns=columns_names)
    phy_maxBW_poly_df = phy_maxBW_poly_df.append(
        {'Bandwidth Required': option_slctd2, 'Bandwidth Sold': y_pred_CIR_Conf_Max_round}, ignore_index=True)
    concatframe_poly = [predictionTbl_poly_df.tail(1), phy_maxBW_poly_df]
    added_poly_df = pd.concat(concatframe_poly)
#/////////////////////////////////////////////////////////////////////////
   # ---------------------------------------------------------------------
    # columns_names = ["Bandwidth Required", "Bandwidth Sold"]
    # phy_maxBW_df = pd.DataFrame(columns=columns_names)
    # phy_maxBW_df = phy_maxBW_df.append(
    #     {'Bandwidth Required': option_slctd2, 'Bandwidth Sold': y_pred_CIR_Conf_Max_round}, ignore_index=True)
    # concatframe = [predictionTbl_df.tail(1), phy_maxBW_df]
    # added_df = pd.concat(concatframe)

    # adjusting the area plot for forcast graph
 
    past_value_df=(cir_conf_time_df.tail(1))
    past_value_df.rename(columns={"CIR_Conf": "predicted"}, inplace=True)
    past_value_df.index.names = ["date"]
    framesArea = [past_value_df, planning_BW_acquisition]
    area_plot_focast_df = pd.concat(framesArea)
    # print("-----")
    # print(past_value_df)
    # print("---")
    # print(planning_BW_acquisition)
    # print("---")
    # print(area_plot_focast_df)
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
    trace1 = px.violin(data_frame=datasetCopy, x="CIR_Conf", y="CIR_Alloc", orientation="v", box=True, color='CIR_Conf', log_x=True, log_y=True, labels={"CIR_Conf": "Configured", "CIR_Alloc": "CIR Allocated"}, width=730, height=400, template='plotly_dark', animation_frame='CIR_Conf', range_x=[10, 300], range_y=[5, 70],).update_layout(yaxis=dict(tickfont=dict(size=1)), xaxis=dict(tickfont=dict(size=1)), font=dict(family="Courier New, monospace", size=10, color="white"), legend=dict(
        x=0, y=1, title_font_family="Times New Roman", font=dict(family="Courier", size=12, color="white"), bgcolor=None, bordercolor=None, borderwidth=0)).add_trace(px.line(dffLine, x="CIR-Configured", y="percentile").update_traces(mode='markers+lines').update_layout(title='', title_x=0.5).data[0])

    trace2 = px.line(data_frame=dffLine, x="CIR-Configured", y="CIR-Configured", template='presentation', width=1490, height=600, labels={"CIR-Configured": "Bandwidth Sold", "percentile": "Bandwidth Required" + str(percentil)}).update_traces(
        mode='markers+lines').add_trace(px.area(data_frame=dffLine, x="CIR-Configured", y="percentile").data[0]).add_trace(px.scatter(dffLine, x=dffLine["CIR-Configured"], y=dffLine["percentile"], text=dffLine["diff"]).data[0]).add_trace(px.line(data_frame=dffLine, x="CIR-Configured", y="CIR-Configured").data[0]).update_layout(yaxis=dict(tickfont=dict(size=12)), xaxis=dict(tickfont=dict(size=12)), font=dict(family="Courier New, monospace", size=10, color="black")).add_trace(px.scatter(data_frame=dffLine, x="CIR-Configured", y="CIR-Configured", text="CIR-Configured").data[0]).update_traces(textposition="top right", hovertemplate=None, hoverinfo='skip').add_trace(px.scatter(data_frame=dffLine, x="CIR-Configured", y="percentile", labels={"CIR-Configured": "Bandwidth Sold", "percentile": "Bandwidth Required"}).data[0])

    trace3 = px.line(data_frame=predictionTbl_poly_df, x="Bandwidth Required", y="Bandwidth Sold", labels={"Bandwidth Required": "Bandwidth Required" + "(" + str(percentil)+"%)"}, template='presentation', width=1000, height=600).update_traces(mode='markers+lines').add_trace(px.area(data_frame=added_poly_df, x="Bandwidth Required", y="Bandwidth Sold").data[0]).update_traces(textposition="bottom right", fillcolor='#32a852').add_trace(px.scatter(data_frame=added_poly_df, x="Bandwidth Required", y="Bandwidth Sold", text="Bandwidth Sold").data[0]).update_traces(textposition="bottom right", fillcolor='#32a852', textfont=dict(family="sans serif", size=18, color="blue")).add_trace(px.scatter(
        data_frame=predictionTbl_poly_df[:-1], x="Bandwidth Required", y="Bandwidth Sold", text="Bandwidth Sold").data[0]).update_traces(textposition="bottom right").update_layout(title='', title_x=0.5, showlegend=False,).update_layout(yaxis=dict(tickfont=dict(size=12)), xaxis=dict(tickfont=dict(size=12)), font=dict(family="Courier New, monospace", size=10, color="black"))

    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
    trace4 = px.pie(pie_df, values=[max_cr_conf, left], names=['Bandwidth Sold', 'Left'], template='presentation', title='').update_traces(
        hoverinfo='label+percent', textinfo='percent+value', textfont_size=20, marker=dict(colors=colors, line=dict(color='#000000', width=2)), pull=(0, 0.1))

    trace5 = px.line(data_frame=cir_conf_time_df, x=cir_conf_time_df.index, y="CIR_Conf", template='plotly_dark', width=1490, height=500, labels={"CIR_Conf": "Bandwidth Sold"}).update_traces(mode='markers+lines').add_trace(px.area(data_frame=area_plot_focast_df, x=area_plot_focast_df.index, y="predicted").data[0]).update_traces(textposition="bottom right", fillcolor='#32a852').add_trace(px.line(data_frame=planning_BW_acquisition_up_to_max).data[0]).add_trace(px.line(data_frame=sales_forecast_df, x=sales_forecast_df.index, y="predicted", text="predicted").data[0]).update_layout(
        yaxis=dict(tickfont=dict(size=12)),
        xaxis=dict(tickfont=dict(size=12)),
        font=dict(family="Courier New, monospace", size=12, color="yellow"))

    trace6 = px.line(data_frame=plot_dff, x="Time", y=["Sold", "Requested", "Required" + "(" + str(percentil)+"%)"], template='plotly_dark', width=760, height=400).update_layout(
        yaxis=dict(tickfont=dict(size=12)),
        xaxis=dict(tickfont=dict(size=10)),
        # xaxis=dict(tickfont=dict(size=8), visible=False),
        # legend_title=None,
        font=dict(family="Courier New, monospace", size=10, color="yellow"), legend=dict(
            x=0, y=1, title_font_family="Times New Roman",
            font=dict(family="Courier", size=12, color="white"), bgcolor=None,
            bordercolor=None, borderwidth=0))
    return str(max_cr_conf)+" mbps", RT, Overbooking_Ratio, equation, str(y_pred_CIR_Conf_Max_round)+" mbps", str(left)+" mbps", trace1, trace3, trace4, predictionTbl_df[["Bandwidth Sold", "Bandwidth Required"]].to_dict('records'), planning_BW_acquisition_value, trace5, trace6, trace2


if __name__ == '__main__':
    app.run_server(debug=False, port=8000)

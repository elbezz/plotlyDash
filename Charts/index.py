import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import pandas_datareader.data as web
import datetime


account_voucher_df = pd.read_csv(r'C:\Users\NextG\account_voucher.csv', usecols=[
                                 "id", "create_uid", "reference", "date", "partner_id", "state", "type", "amount"])
vProduct_df = pd.read_csv(r'C:\Users\NextG\vproduct.csv', usecols=[
                          "id", "name_template", "productcategoryname"])
vPartner_df = pd.read_csv(r'C:\Users\NextG\res_partner.csv', usecols=[
                          "id", "name", "active", "zip", "city_moved1", "city"])
vuser_df = pd.read_csv(r'C:\Users\NextG\res_users.csv',
                       usecols=["id", "active", "login"])

account_voucher_df.rename(columns={'id': 'voucher_id'}, inplace=True)
vPartner_df.rename(columns={'id': 'partner_id'}, inplace=True)
vPartner_df.rename(columns={'name': 'partner'}, inplace=True)
account_voucher_df.rename(columns={'create_uid': 'login_id'}, inplace=True)
vuser_df.rename(columns={'id': 'login_id'}, inplace=True)

merge1 = account_voucher_df[["voucher_id", "login_id", "reference", "date", "partner_id",
                             "state", "type", "amount"]].merge(vuser_df[['login_id', 'login']], on="login_id")
merge2 = merge1[["voucher_id", "login_id", "reference", "date", "partner_id", "state",
                 "type", "amount", "login"]].merge(vPartner_df[["partner_id", "partner"]], on="partner_id")
merge2['date'] = pd.to_datetime(merge2.date)

merge2['year'] = pd.DatetimeIndex(merge2.date).year
merge2['monthNum'] = pd.DatetimeIndex(merge2.date).month
merge2['dayNum'] = pd.DatetimeIndex(merge2.date).day
merge2['weekday'] = pd.DatetimeIndex(merge2.date).weekday
merge2['day'] = merge2['date'].dt.day_name()
merge2['month'] = merge2['date'].dt.month_name()

voucher = merge2

voucher.drop(['voucher_id', 'partner_id', 'login_id'], axis=1, inplace=True)





app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

app.layout = html.Div([
    html.Div([
        dbc.Row(
            dbc.Col(html.H1("Sales Dashboard",className='text-center text-primary mb-4'),
                width=12)
        
        ),
        
        ]),
    
    
        
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="slct_year",
                                 options=[
                                     {"label": "2015", "value": 2015},
                                     {"label": "2016", "value": 2016},
                                     {"label": "2017", "value": 2017},
                                     {"label": "2018", "value": 2018},
                                     {"label": "2019", "value": 2019},
                                     {"label": "2020", "value": 2020},
                                     {"label": "2021", "value": 2021}],
                    multi=False,
                    value=2020,
            ),
                width={"size": 3,  "offset": 2, "order": 3}),
    
    
    
             dbc.Col(dcc.Dropdown(id="slct_month",
                                 options=[
                                     {"label": "January", "value": "January"},
                                     {"label": "February", "value": "February"},
                                     {"label": "March", "value": "March"},
                                     {"label": "April", "value": "April"},
                                     {"label": "May", "value": "May"},
                                     {"label": "June", "value": "June"},
                                     {"label": "July", "value": "July"},
                                     {"label": "August", "value": "August"},
                                     {"label": "September", "value": "September"},
                                     {"label": "October", "value": "October"},
                                     {"label": "November", "value": "November"},
                                     {"label": "December", "value": "December"}],
                    multi=True,
                    value="May",
            ),
                     
                    width={'size': 3, "offset": 2, 'order': 3})]),       
        
     
    
        dbc.Row(
        [
           dbc.Col(dcc.Graph(id='voucher_bar', figure={}),
           width=8, lg={'size': 3,  "offset": 2, 'order': 'first'}
           ),
        ])
    
    
        
    ])
    
@app.callback(

    dash.dependencies.Output(component_id='voucher_bar',
                             component_property='figure'),
    [dash.dependencies.Input(component_id='slct_year', component_property='value'),
     dash.dependencies.Input(component_id='slct_month',
                             component_property='value')
     ]
)
def update_graph(option_slctd, option_slctd2):

    dff = voucher.copy()
    dff = dff[dff["year"] == option_slctd]
    dff = dff[dff["month"] == option_slctd2]

    # Plotly Express
    fig = px.bar(
        dff, x="login", y="amount", barmode="group",

        # template='plotly_dark'
    )
    return fig




if __name__ == '__main__':
    app.run_server(debug=False)

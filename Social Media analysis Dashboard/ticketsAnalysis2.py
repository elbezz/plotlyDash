from logging import disable
import dash                              # pip install dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import dash_table
from dash_extensions import Lottie       # pip install dash-extensions
# pip install dash-bootstrap-components
import dash_bootstrap_components as dbc
import plotly.express as px              # pip install plotly
import pandas as pd                      # pip install pandas
from datetime import date
import calendar
from wordcloud import WordCloud
#################################################################################
# import dataframes from csv
ticket_df = pd.read_csv(r'D:\data\tblticket.csv')
ticketReply_df= pd.read_csv(r'D:\data\tblticketreplies.csv')
tbladmins_df = pd.read_csv(r'D:\data\tbladmins.csv')
tbldepartment_df = pd.read_csv(r'D:\data\tbldepartment.csv')
employee_df = pd.read_csv(r'D:\data\employee.csv')
tblclient_df = pd.read_csv(r'D:\data\tblclient.csv')
tblticketfeedback_df = pd.read_csv(r'D:\data\tblticketfeedback.csv')

ticketReply_firstReply_df=ticketReply_df.groupby(['tid']).agg(first_reply_time=('date','min'))


filterNotNulluserid = ticket_df.userid!=0
ticket_df = ticket_df[filterNotNulluserid]

ticketReply_df.admin=ticketReply_df.admin.fillna(0)
filterNotNaNAdmin = ticketReply_df.admin!=0
ticketReply_df = ticketReply_df[filterNotNaNAdmin]

ticket_df['lastreply'] = pd.to_datetime(ticket_df.lastreply)
ticket_df['date'] = pd.to_datetime(ticket_df.date)
ticket_df['time_to_resolve']=(ticket_df.lastreply-ticket_df.date)

ticketReply_df=pd.merge(ticketReply_df, ticketReply_firstReply_df,how='left', on="tid")
ticketReply_df=pd.merge(ticketReply_df, ticket_df[['tid','date','time_to_resolve']],how='left', on="tid")

ticketReply_df.rename(columns={'date_x':'date_reply'}, inplace=True)
ticketReply_df.rename(columns={'date_y':'date_ticket'}, inplace=True)

ticketReply_df['date_ticket'] = pd.to_datetime(ticketReply_df.date_ticket)
ticketReply_df['date_reply'] = pd.to_datetime(ticketReply_df.date_reply)
ticketReply_df['first_reply_time'] = pd.to_datetime(ticketReply_df.first_reply_time)

ticketReply_df['date_as_key'] = ticketReply_df['date_reply']
ticketReply_df['date_as_key'] = pd.to_datetime(ticketReply_df['date_as_key']).dt.date


ticketReply_df['first_reply_time_duration']= (ticketReply_df.first_reply_time-ticketReply_df.date_ticket)



employee_df.rename(columns={'name_related':'admin'}, inplace=True)

tbldepartment_df.rename(columns={'id':'department_id'}, inplace=True)

employee_df=pd.merge(employee_df, tbldepartment_df[['department_id','name']],how='left', on="department_id")

employee_df.drop(['department_id'], axis=1,inplace=True)

employee_df.rename(columns={'work_email':'email'}, inplace=True)

employee_df.rename(columns={'name':'department_name'}, inplace=True)

tbladmins_df['email']=tbladmins_df['email'].str.strip()
employee_df['email']=employee_df['email'].str.strip()

tbladmins_df=pd.merge(tbladmins_df, employee_df[['email','department_name']],how='left', on="email")



ticketReply_df['admin']=ticketReply_df['admin'].str.strip()
tbladmins_df['admin']=tbladmins_df['admin'].str.strip()

ticketReply_df=pd.merge(ticketReply_df, tbladmins_df[['admin','department_name']],how='left', on="admin")

tbladmins_df.rename(columns={'id':'adminid'}, inplace=True)

tblticketfeedback_df=pd.merge(tblticketfeedback_df, tbladmins_df[['adminid','admin','department_name']],how='left', on="adminid")

tblticketfeedback_df['date'] = pd.to_datetime(tblticketfeedback_df['date']).dt.date

ticketReply_df["date_as_key"] = pd.to_datetime(ticketReply_df["date_as_key"])
ticketReply_df["month"] = ticketReply_df["date_as_key"].dt.month
ticketReply_df['month'] = ticketReply_df['month'].apply(lambda x: calendar.month_abbr[x])

ticketReply_df["year"] = ticketReply_df["date_as_key"].dt.year
ticketReply_df["day"] = ticketReply_df["date_as_key"].dt.day

gtigtr_df= ticketReply_df.drop_duplicates(['tid'])
gtigtr_df=gtigtr_df.dropna(subset=['date_ticket'])
print('-------------')
print(gtigtr_df)

ticketReply_replyPerTicket=ticketReply_df.groupby(['tid','admin','department_name']).agg(reply=('id','count'))

ticketReply_replyTicketAdmin=ticketReply_df.groupby(['admin','date_as_key','department_name'], as_index=False).agg({"id": pd.Series.count, "tid": pd.Series.nunique})

ticketReply_replyTicketAdmin["month"] = ticketReply_replyTicketAdmin["date_as_key"].dt.month
ticketReply_replyTicketAdmin["year"] = ticketReply_replyTicketAdmin["date_as_key"].dt.year

ticketReply_replyTicketAdmin.rename(columns={'id':'numOfResponses','tid':'numOfTickets'}, inplace=True)
ticketReply_replyTicketAdmin['month'] = ticketReply_replyTicketAdmin['month'].apply(lambda x: calendar.month_abbr[x])
ticketbyDep_df=ticketReply_replyTicketAdmin.groupby(['department_name','month','year'], as_index=False).agg({"numOfResponses": pd.Series.sum, "numOfTickets": pd.Series.sum})
ticketbyDep_df['efficacy']=round((ticketbyDep_df.numOfResponses/ticketbyDep_df.numOfTickets),2)
adminTicketRes_df=ticketReply_replyTicketAdmin.groupby(['admin','month','year','department_name'], as_index=False).agg({"numOfResponses": pd.Series.sum, "numOfTickets": pd.Series.sum})
adminTicketRes_df['efficacy']=round((adminTicketRes_df.numOfResponses/adminTicketRes_df.numOfTickets),2)
# print(ticketReply_replyTicketAdmin)
# ticketbyDep_df['month'] = ticketbyDep_df['month'].apply(lambda x: calendar.month_abbr[x])


tblticketfeedback_df["date"] = pd.to_datetime(tblticketfeedback_df["date"])
tblticketfeedback_df["month"] = tblticketfeedback_df["date"].dt.month
tblticketfeedback_df["year"] = tblticketfeedback_df["date"].dt.year
tblticketfeedback_df['month'] = tblticketfeedback_df['month'].apply(lambda x: calendar.month_abbr[x])
feedback_df= tblticketfeedback_df[['ticketid','admin','rating','comments','date','month','year','department_name']]
feedback_df['date'] = feedback_df['date'].dt.date
adminRating_df=feedback_df.groupby(['admin','month','year','department_name'], as_index=False).agg({"rating": pd.Series.mean})

ticketoverview=ticketReply_replyTicketAdmin.groupby(['month','year'], as_index=False).agg({"numOfResponses": pd.Series.sum, "numOfTickets": pd.Series.sum})
ticket_df["month"] = ticket_df["date"].dt.month
ticket_df["year"] = ticket_df["date"].dt.year
ticket_df['month'] = ticket_df['month'].apply(lambda x: calendar.month_abbr[x])
filterspam = ticket_df.time_to_resolve==pd.Timedelta("0 days 0 hours")
spam_df = ticket_df[filterspam]
spam_df=spam_df.groupby(['month','year'], as_index=False).agg({"time_to_resolve": pd.Series.count})
#######3333 print(spam_df)
#################################################################################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container([


    
    dbc.Row([
            # dbc.Col([
            #     dbc.Card([
                    
            #         dbc.CardBody([

            #             dbc.CardImg(src="/static/images/ayrade.jpg", top=True),

            #         ], style={'textAlign': 'center'})

            #     ], color="light", outline=True),

            # ], width=2,style={"margin-right": "-10px"}),


            dbc.Col([
                dbc.Card([
                    
                    dbc.CardBody([
                    #     dcc.DatePickerSingle(
                    #     id='my-date-picker-start',
                    #     date=date(2021, 1, 1),
                    #     display_format='YYYY,MM,DD',
                    #     className='align-self-center mb-3 mt-0'
                    # ),
                    dcc.Dropdown(id="slct_year",
                                 options=[
                                     {"label": "2015", "value": 2015},
                                     {"label": "2016", "value": 2016},
                                     {"label": "2017", "value": 2017},
                                     {"label": "2018", "value": 2018},
                                     {"label": "2019", "value": 2019},
                                     {"label": "2020", "value": 2020},
                                     {"label": "2021", "value": 2021}],
                    multi=False,
                    value="",
            ),
                 
                dcc.Dropdown(id="slct_month",
                                 options=[
                                     {"label": "January", "value": "Jan"},
                                     {"label": "February", "value": "Feb"},
                                     {"label": "March", "value": "Mar"},
                                     {"label": "April", "value": "Apr"},
                                     {"label": "May", "value": "May"},
                                     {"label": "June", "value": "Jun"},
                                     {"label": "July", "value": "Jul"},
                                     {"label": "August", "value": "Aug"},
                                     {"label": "September", "value": "Sep"},
                                     {"label": "October", "value": "Oct"},
                                     {"label": "November", "value": "Nov"},
                                     {"label": "December", "value": "Dec"}],
                    multi=False,
                    value="",
            ),                    

                    ],)

                ], style={"border-radius":"2%","background":"primary", 'outline':'True','textAlign': 'center','margin-top': '20px','height': '120px'}),

            ], width=2),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Checklist(
                           id='depChecklist', 
                            options=[
                                {'label': 'SI', 'value': 'SI'},
                                {'label': 'Project', 'value': 'Project'},
                                {'label': 'Sales', 'value': 'Sales'}
                                ],
                           value=['SI'],
                           style={"display": "block"},
                           labelStyle={"display": "block"}
                        )

                    ], 
                   
                    )

                ], style={"border-radius":"2%","background":"light",'height': '120px','margin-top': '20px'}),

            ], width=2),

          
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                    html.H3('GTI'),
                    html.H5(id='content-gti', children="000")
                    ],style={"border-radius":"8%","background":'rgba(50, 250, 120, 1.0)', 'inverse':'True','textAlign': 'center','margin-top': '20px','height': '120px'})

                ], color="light", outline=True),

            ], width=2),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                    html.H3('GTR'),
                    html.H5(id='content-gtr', children="000")
                    ], style={'textAlign': 'center'})

                ], style={"border-radius":"8%","background":'rgba(190, 255, 255, 1.0)','textAlign': 'center','margin-top': '20px','height': '120px'}),

            ], width=2),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                    html.H6('Efficacité'),
                    html.Br(),
                    html.H3(id='content-efficacite', children="000")
                    ], style={'textAlign': 'center'})

                ], style={"border-radius":"8%","background":"PowderBlue",'textAlign': 'center','margin-top': '20px','height': '120px'}),

            ], width=2),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                    html.H6('évaluation'),
                    html.Br(),
                    html.H3(id='content-evalClient', children="000")
                    ], style={"border-radius":"8%","background":"PowderBlue",'textAlign': 'center','margin-top': '20px','height': '120px'})

                ], color="light", outline=True),

            ], width=2),
            ], className='mb-2 mt-2'),

    dbc.Row([
            dbc.Col([
                dbc.Card([

                    dbc.CardBody([
                    dcc.Graph(id='bar-chart-ticket-response', figure={}),
                    ], style={'textAlign': 'center'})

                ], color="light", outline=True),

            ], width=4),
            dbc.Col([
                dbc.Card([

                    dbc.CardBody([
                    dcc.Graph(id='bar-chart-efficacy-dep', figure={}),
                    ], style={'textAlign': 'center'})

                ], color="light", outline=True),

            ], width=4),
        dbc.Col([
            dbc.Card([

                dbc.CardBody([
                    dcc.Graph(id='bar-chart-all-services', figure={}),

                ], style={'textAlign': 'center'})

            ], color="light", outline=True),

        ], width=4),
    ]),

    dbc.Row([

        dbc.Col([
                dbc.Card([

                dbc.CardBody([
                    dcc.Graph(id='bar-chart-Tickets-traités-réponses-par-Admin', figure={}),
                ])

                ], color="light", outline=True),

                ], width=6),

        dbc.Col([
                dbc.Card([

                dbc.CardBody([
                    dcc.Graph(id='bar-chart-efficacy-Admin', figure={}),
                ])

                ], color="light", outline=True),

                ], width=6),

            ]),
            
    
    dbc.Row([

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                    dcc.Graph(id='bar-chart_feedback', figure={}),
                    ], style={'textAlign': 'center'})

                ], color="light", outline=True),

            ], width=6),

             dbc.Col([
            dbc.Card([

                dbc.CardBody([
                    html.H6('feedback détail'),
                    dash_table.DataTable(id='feedbacktable',
                    columns=[{'name': i,'id': i} for i in feedback_df.columns],
                    data=feedback_df.to_dict('records'),

                            style_header={
                                    'backgroundColor': 'black',
                                    'fontWeight': 'bold',
                                    'color': 'white',
                                    'border': '5px',
                                    'font_family': 'Arial',
                                    'font_size' : '12px',
                                    'line_height': '2px',
                                    # 'whiteSpace': 'normal',
                                },
                                    style_cell={'padding-left': '35px','padding-right': '35px', 'border': '0px', 'textAlign': 'left', 'font_family': 'Roboto', 'fontWeight': '100', 'font_size' : '14px', 'line_height': '5px'},
                                    style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': 'light_grey',
                                    }
                                    ],
                                    # style_cell_conditional=[
                                    # {
                                    # 'if': {'column_id': c},
                                    #     'textAlign': 'left'
                                    # } for c in ['']
                                    # ],
                                    style_as_list_view=True,
                                    page_action='native',
                                    fixed_rows={'headers':True},
                                    style_table={'height': '600px', 'overflowY': 'auto'},
                    # editable=True,
                    # filter_action="native",
                    # sort_action="native",
                    # sort_mode="multi",
                    # column_selectable="single",
                    # row_selectable="multi",
                    # row_deletable=True,
                    # selected_columns=[],
                    # selected_rows=[],
                    # page_action="native",
                    # page_current= 0,
                    # page_size= 10,
                    ),

                ],)

            ], color="light", outline=True),

        ], width=6),



        ]),





], fluid=True)

#Updating the 5 number cards ******************************************
@app.callback(
    Output('content-gti','children'),
    Output('content-gtr','children'),
    Output('content-efficacite','children'),
    Output('content-evalClient','children'),
    Input('slct_year', 'value'),
    Input('slct_month', 'value'),
    Input('depChecklist', 'value')
)

def update_small_cards(option_slctd, option_slctd2,option_slcdep):

    
    gtigtr = gtigtr_df.copy()
    gtigtr = gtigtr[gtigtr["year"] == option_slctd]
    gtigtr = gtigtr[gtigtr["month"] == option_slctd2]
    gtigtr = gtigtr[gtigtr["department_name"].isin(option_slcdep)]
    # gtr
    time_to_resolve = gtigtr["time_to_resolve"].mean()
    # gti
    first_reply_time_duration = gtigtr["first_reply_time_duration"].mean()

    def format_timedelta_to_HHMMSS(td):
        td_in_seconds = td.total_seconds()
        hours, remainder = divmod(td_in_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        hours = int(hours)
        minutes = int(minutes)
        seconds = int(seconds)
        if minutes < 10:
            minutes = "0{}".format(minutes)
        if seconds < 10:
            seconds = "0{}".format(seconds)
        return "{}:{}:{}".format(hours, minutes,seconds)

    
    gti=format_timedelta_to_HHMMSS(first_reply_time_duration)
    gtr=format_timedelta_to_HHMMSS(time_to_resolve)

    #efficacite
    ticketbyDep = ticketbyDep_df.copy()
    ticketbyDep = ticketbyDep[ticketbyDep["year"] == option_slctd]
    ticketbyDep = ticketbyDep[ticketbyDep["month"] == option_slctd2]
    ticketbyDep = ticketbyDep[ticketbyDep["department_name"].isin (option_slcdep)]
    numRep = ticketbyDep["numOfResponses"].mean()
    numTick = ticketbyDep["numOfTickets"].mean()
    effica= numRep/numTick
    effica = round(effica,2)

    # evaluation
    tblticketfeedback = tblticketfeedback_df.copy()
    tblticketfeedback = tblticketfeedback[tblticketfeedback["year"] == option_slctd]
    tblticketfeedback = tblticketfeedback[tblticketfeedback["month"] == option_slctd2]
    tblticketfeedback = tblticketfeedback[tblticketfeedback["department_name"].isin (option_slcdep)]
    tblticketfeedback = tblticketfeedback["rating"].mean()
    tblticketfeedback = round(tblticketfeedback,2)


    return gti,gtr,effica, tblticketfeedback


# Bar Chart1 ************************************************************
@app.callback(
    Output('bar-chart-Tickets-traités-réponses-par-Admin', 'figure'),
    Input('slct_year', 'value'),
    Input('slct_month', 'value'),
    Input('depChecklist', 'value')
    )

def update_graph(option_slctd, option_slctd2,option_slcdep):

    dff = adminTicketRes_df.copy()
    dff = dff[dff["year"] == option_slctd]
    dff = dff[dff["month"] == option_slctd2]
    dff = dff[dff["department_name"].isin (option_slcdep)]


    trace1 = go.Bar(x=dff.admin, y=dff[('numOfResponses')], name='Response',marker=dict(color='rgba(120, 120, 255, 1.0)'))
    trace2 = go.Bar(x=dff.admin, y=dff[('numOfTickets')], name='Tickets',marker=dict(color='rgba(50, 160, 80, 1.0)'))


    # marker=dict(
    #     color='rgba(246, 78, 139, 0.6)',
    #     line=dict(color='rgba(246, 78, 139, 1.0)', width=3)


    return {
        'data': [trace1, trace2],
        'layout':
        go.Layout(
            title='Tickets traités/réponses par Admin',
            barmode='group',
            xaxis={'categoryorder':'total descending'}
             )
    }

@app.callback(
    Output('bar-chart-efficacy-Admin', 'figure'),
    Input('slct_year', 'value'),
    Input('slct_month', 'value'),
    Input('depChecklist', 'value')
    )

def update_graph(option_slctd, option_slctd2,option_slcdep):

    dff = adminTicketRes_df.copy()
    dff = dff[dff["year"] == option_slctd]
    dff = dff[dff["month"] == option_slctd2]
    dff = dff[dff["department_name"].isin (option_slcdep)]


    trace = go.Bar(x=dff.admin, y=dff[('efficacy')], name='Response',marker=dict(color='rgba(250, 230, 70, 1.0)'))


    # marker=dict(
    #     color='rgba(246, 78, 139, 0.6)',
    #     line=dict(color='rgba(246, 78, 139, 1.0)', width=3)


    return {
        'data': [trace],
        'layout':
        go.Layout(
            title='efficacy Admin',
            barmode='group',
            xaxis={'categoryorder':'total ascending'}
             )
    }

@app.callback(
    Output('bar-chart-ticket-response', 'figure'),
    Input('slct_year', 'value'),
    Input('slct_month', 'value'),
    )

def update_graph(option_slctd,option_slctd2):

    dff = ticketoverview.copy()
    dff = dff[dff["year"] == option_slctd]
    dff = dff[dff["month"] == option_slctd2]

    dff2 = spam_df.copy()
    dff2 = dff2[dff2["year"] == option_slctd]
    dff2 = dff2[dff2["month"] == option_slctd2]

    trace1 = go.Bar(x=dff.month, y=dff[('numOfResponses')], name='Response',marker=dict(color='rgba(120, 120, 255, 1.0)'))
    trace2 = go.Bar(x=dff.month, y=dff[('numOfTickets')], name='Tickets',marker=dict(color='rgba(50, 160, 80, 1.0)'))
    trace3 = go.Bar(x=dff2.month, y=dff2[('time_to_resolve')], name='Spams',marker=dict(color='rgba(250, 10, 10, 1.0)'))

    # marker=dict(
    #     color='rgba(246, 78, 139, 0.6)',
    #     line=dict(color='rgba(246, 78, 139, 1.0)', width=3)


    return {
        'data': [trace1,trace2,trace3],
        'layout':
        go.Layout(
            title='Tickets/réponses overview',
            barmode='group'
             )
    }

# Bar Chart ************************************************************
@app.callback(
    Output('bar-chart-all-services', 'figure'),
    Input('slct_year', 'value'),
    Input('slct_month', 'value')
    )

def update_graph(option_slctd, option_slctd2):

    dff = ticketbyDep_df.copy()
    dff = dff[dff["year"] == option_slctd]
    dff = dff[dff["month"] == option_slctd2]

    trace1 = go.Bar(x=dff.department_name, y=dff[('numOfResponses')], name='Response',marker=dict(color='rgba(120, 120, 255, 1.0)'))
    trace2 = go.Bar(x=dff.department_name, y=dff[('numOfTickets')], name='Tickets',marker=dict(color='rgba(50, 160, 80, 1.0)'))

    return {
        'data': [trace1, trace2],
        'layout':
        go.Layout(
            title='Tickets/réponses vs départ',
            barmode='group',
            xaxis={'categoryorder':'total ascending'}
            )
    }


@app.callback(
    Output('bar-chart-efficacy-dep', 'figure'),
    Input('slct_year', 'value'),
    Input('slct_month', 'value')
    )

def update_graph(option_slctd, option_slctd2):

    dff = ticketbyDep_df.copy()
    dff = dff[dff["year"] == option_slctd]
    dff = dff[dff["month"] == option_slctd2]

    trace = go.Bar(x=dff.efficacy, y=dff[('department_name')], name='Response',orientation='h',marker=dict(color='rgba(250, 230, 70, 1.0)'))
    

    return {
        'data': [trace],
        'layout':
        go.Layout(
            title='efficacity par départ ',
            barmode='group',
            yaxis={'categoryorder':'total descending'}
            )
    }

@app.callback(
    Output('bar-chart_feedback', 'figure'),
    Input('slct_year', 'value'),
    Input('slct_month', 'value'),
    Input('depChecklist', 'value')
    )

def update_graph(option_slctd, option_slctd2,option_slcdep):

    dff = adminRating_df.copy()
    dff = dff[dff["year"] == option_slctd]
    dff = dff[dff["month"] == option_slctd2]
    dff = dff[dff["department_name"].isin (option_slcdep)]

    trace1 = go.Bar(x=dff.admin, y=dff[('rating')], name='rating',textposition='auto',orientation='v')


    return {
        'data': [trace1],
        'layout':
        go.Layout(
            title='Partners Rating /10',
            barmode='stack', 
            xaxis={'categoryorder':'total ascending'}
            )

    }

@app.callback(
    Output(component_id='feedbacktable', component_property='data'),
    [Input(component_id='slct_year', component_property='value')],
    [Input(component_id='slct_month', component_property='value')],
    [Input(component_id='depChecklist', component_property='value')]
)
def update_df_div(option_slctd, option_slctd2,option_slcdep):

    dff = feedback_df.copy()
    dff = dff[dff['year'] == option_slctd]
    dff = dff[dff["month"] == option_slctd2]
    dff = dff[dff["department_name"].isin (option_slcdep)]
    data = dff.to_dict('records')
    return data


# ***********************************************************************
# @app.callback(
#     Output('graph-with-slider', 'figure'),
#     Input('year-slider', 'value'))
# def update_figure(selected_year):
#     filtered_df = ticketReply_replyTicketAdmin[ticketReply_replyTicketAdmin.year == selected_year]
#     numTicket=filtered_df['numOfTickets'].sum()
#     numResp=filtered_df['numOfResponses'].sum()
#     fig = px.scatter(filtered_df, x=numTicket, y=numResp,
#                     #  size="numOfTickets", 
#                      color="department_name", hover_name="admin",
#                      log_x=True, size_max=55)

#     fig.update_layout(transition_duration=500)

#     return fig

if __name__ == '__main__':
    app.run_server(debug=False, port=8000)

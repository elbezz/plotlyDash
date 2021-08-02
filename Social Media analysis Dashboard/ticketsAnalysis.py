import dash                              # pip install dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input

from dash_extensions import Lottie       # pip install dash-extensions
# pip install dash-bootstrap-components
import dash_bootstrap_components as dbc
import plotly.express as px              # pip install plotly
import pandas as pd                      # pip install pandas
from datetime import date
import calendar
from wordcloud import WordCloud
#################################################################################
# import dataframes
ticket_df = pd.read_csv(r'D:\data\tblticket.csv')
ticketReply_df= pd.read_csv(r'D:\data\tblticketreplies.csv')
tbladmins_df = pd.read_csv(r'D:\data\tbladmins.csv')
tbldepartment_df = pd.read_csv(r'D:\data\tbldepartment.csv')
employee_df = pd.read_csv(r'D:\data\employee.csv')
tblclient_df = pd.read_csv(r'D:\data\tblclient.csv')
tblticketfeedback_df = pd.read_csv(r'D:\data\tblticketfeedback.csv')

ticketReply_firstReply_df=ticketReply_df.groupby(['tid']).agg(first_reply_time=('date','min'))

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

ticketReply_df.admin=ticketReply_df.admin.fillna(0)
filterNotNaNAdmin = ticketReply_df.admin!=0
ticketReply_df = ticketReply_df[filterNotNaNAdmin]

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

ticketReply_replyPerTicket=ticketReply_df.groupby(['tid','admin','department_name']).agg(reply=('id','count'))

ticketReply_replyTicketAdmin=ticketReply_df.groupby(['admin','date_as_key','department_name'], as_index=False).agg({"id": pd.Series.count, "tid": pd.Series.nunique})

ticketReply_replyTicketAdmin["month"] = ticketReply_replyTicketAdmin["date_as_key"].dt.month
ticketReply_replyTicketAdmin["year"] = ticketReply_replyTicketAdmin["date_as_key"].dt.year

ticketReply_replyTicketAdmin.rename(columns={'id':'numOfResponses','tid':'numOfTickets'}, inplace=True)

ticketbyDep=ticketReply_replyTicketAdmin.groupby(['department_name','month','year'], as_index=False).agg({"numOfResponses": pd.Series.sum, "numOfTickets": pd.Series.sum})

ticketbyDep['month'] = ticketbyDep['month'].apply(lambda x: calendar.month_abbr[x])

print(ticketbyDep)
#################################################################################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container([
    dbc.Row([
            dbc.Col([
                dbc.Card([
                    
                    dbc.CardBody([

                        dbc.CardImg(src="/static/images/ayrade.jpg", top=True),

                    ], style={'textAlign': 'center'})

                ], color="light", outline=True),

            ], width=2,style={"margin-right": "-10px"}),


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
                 
                    #     dcc.DatePickerSingle(
                    #     id='my-date-picker-end',
                    #     date=date(2021, 6, 1),
                    #     display_format='YYYY,MM,DD',
                    #     className='align-self-center'
                    # ),
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

                ], color="light", outline=True),

            ], width=2),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                    html.H3('GTI'),
                    html.H5(id='content-gti', children="000")
                    ], style={'textAlign': 'center'})

                ], color="light", outline=True),

            ], width=2, style={'margin-top': '20px'}),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                    html.H3('GTR'),
                    html.H5(id='content-gtr', children="000")
                    ], style={'textAlign': 'center'})

                ], color="light", outline=True),

            ], width=2,style={'margin-top': '20px'}),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                    html.H6('Efficacité'),
                    html.H5(id='content-efficacite', children="000")
                    ], style={'textAlign': 'center'})

                ], color="light", outline=True),

            ], width=2,style={'margin-top': '20px'}),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                    html.H6('évaluation-clients'),
                    html.H5(id='content-evalClient', children="000")
                    ], style={'textAlign': 'center'})

                ], color="light", outline=True),

            ], width=2,style={'margin-top': '20px'}),
            ], className='mb-2 mt-2'),

    dbc.Row([
        dbc.Col([
                dbc.Card([

                dbc.CardBody([
                    dcc.Graph(id='line-chart', figure={}),
                ])

                ], color="light", outline=True),

                ], width=8),


            dbc.Col([
                dbc.Card([
                dbc.CardBody([
                    # dcc.Graph(id='bar-chart', figure={}),
                        dcc.Graph(id='graph-with-slider'),
                        dcc.Slider(
                            id='year-slider',
                            min=ticketReply_replyTicketAdmin['year'].min(),
                            max=ticketReply_replyTicketAdmin['year'].max(),
                            value=ticketReply_replyTicketAdmin['year'].min(),
                            marks={str(year): str(year) for year in ticketReply_replyTicketAdmin['year'].unique()},
                            step=None
                        )


                ])

                ], color="light", outline=True),

            ], width=4),
            ]),
    dbc.Row([
            dbc.Col([
                dbc.Card([

                    dbc.CardBody([
                    dcc.Graph(id='line-chart2', figure={}),
                    ], style={'textAlign': 'center'})

                ], color="light", outline=True),

            ], width=8),


            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                    dcc.Graph(id='pie-chart', figure={}),
                    ], style={'textAlign': 'center'})

                ], color="light", outline=True),

            ], width=4),
            ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([

                dbc.CardBody([
                    dcc.Graph(id='bar-chart', figure={}),

                ], style={'textAlign': 'center'})

            ], color="light", outline=True),

        ], width=12),
    ])



])

#Updating the 5 number cards ******************************************
@app.callback(
    Output('content-gti','children'),
    Output('content-gtr','children'),
    # Output('content-efficacite','children'),
    Output('content-evalClient','children'),
    Input('my-date-picker-start','date'),
    Input('my-date-picker-end','date'),
)

def update_small_cards(start_date, end_date):

    # gtr
    ticketReply = ticketReply_df.copy()
    ticketReply = ticketReply[(ticketReply['date_as_key']>=start_date) & (ticketReply['date_as_key']<=end_date)]
    time_to_resolve = ticketReply["time_to_resolve"].mean()

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

    gtr=format_timedelta_to_HHMMSS(time_to_resolve)



        # return f"{hour}:{minutes}:{seconds}"

    # gti
    ticketReply = ticketReply_df.copy()
    ticketReply = ticketReply[(ticketReply['date_as_key']>=start_date) & (ticketReply['date_as_key']<=end_date)]
    first_reply_time_duration = ticketReply["first_reply_time_duration"].mean()
    
    gti=format_timedelta_to_HHMMSS(first_reply_time_duration)
    #efficacite
  
    # dff_i = df_invite.copy()
    # dff_i = dff_i[(dff_i['Sent At']>=start_date) & (dff_i['Sent At']<=end_date)]
    

    # evaluation
    tblticketfeedback = tblticketfeedback_df.copy()
    tblticketfeedback = tblticketfeedback[(tblticketfeedback['date']>=start_date) & (tblticketfeedback['date']<=end_date)]
    tblticketfeedback = tblticketfeedback["rating"].mean()

    return gti,gtr, tblticketfeedback


# Bar Chart ************************************************************
@app.callback(
    Output('bar-chart', 'figure'),
    Input('slct_year', 'value'),
    Input('slct_month', 'value')
    )

def update_graph(option_slctd, option_slctd2):

    dff = ticketbyDep.copy()
    dff = dff[dff["year"] == option_slctd]
    dff = dff[dff["month"] == option_slctd2]
 
    fig = px.bar(
        dff, x="department_name", y=['numOfResponses','numOfTickets'], color='department_name',
              barmode="stack",

         template='plotly_dark'
    )
    return fig
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

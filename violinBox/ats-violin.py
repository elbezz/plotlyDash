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

# dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
# dataset['time'] = dataset['timestamp'].dt.strftime('%H:%M:%S')
# dataset['hour'] = dataset['timestamp'].dt.strftime('%H')
# dataset.drop('timestamp', axis=1, inplace=True)

def percentil95(x):
    return np.percentile(x, 95)


# ------------------------------------------------------------------------------
maximum_df = dataset.groupby(
    'CIR_Conf')['CIR_Alloc'].max().reset_index(name='maximum')
mean_df = dataset.groupby('CIR_Conf')[
    'CIR_Alloc'].mean().reset_index(name='mean')

prcl_df = dataset.groupby('CIR_Conf')['CIR_Alloc'].agg(
    [percentil95]).reset_index()

#--------------------------------------------------------------------------------
# Build the violin/box plot

violinfig = px.violin(
    # data_frame=df.query("State == ['{}','{}']".format('ALABAMA','NEW YORK')),
    data_frame=dataset,
    x="CIR_Conf",
    y="CIR_Alloc",
    # category_orders={'Affected by': [
    #     'Disease', 'Unknown', 'Pesticides', 'Other', 'Pests_excl_Varroa', 'Varroa_mites']},
    orientation="v",              # vertical 'v' or horizontal 'h'
    # points='all',               # 'outliers','suspectedoutliers', 'all', or False
    box=True,                   # draw box inside the violins
    color='CIR_Conf',              # differentiate markers by color
    # violinmode="overlay",       # 'overlay' or 'group'
    # color_discrete_sequence=["limegreen","red"],
    # color_discrete_map={"ALABAMA": "blue" ,"NEW YORK":"magenta"}, # map your chosen colors

    # hover_name='Year',          # values appear in bold in the hover tooltip
    # hover_data=['State'],       # values appear as extra data in the hover tooltip
    # custom_data=['Program'],    # values are extra data to be used in Dash callbacks

    # facet_row='State',          # assign marks to subplots in the vertical direction
    # facet_col='Period',         # assign marks to subplots in the horizontal direction
    # facet_col_wrap=2,           # maximum number of subplot columns. Do not set facet_row

    log_x=True,                 # x-axis is log-scaled
    log_y=True,                 # y-axis is log-scaled
    
    labels={"CIR_Conf": "CIR Configured",
            "CIR_Alloc": "CIR Allocated"},     # map the labels
    # title='What is killing our Bees',
    width=2000,                   # figure width in pixels
    height=600,                   # igure height in pixels
    template='presentation',      # 'ggplot2', 'seaborn', 'simple_white', 'plotly',
                                  # 'plotly_white', 'plotly_dark', 'presentation',
                                  # 'xgridoff', 'ygridoff', 'gridon', 'none'

    animation_frame='CIR_Conf',     # assign marks to animation frames
    # animation_group='',         # use only when df has multiple rows with same object
    range_x=[10,300],             # set range of x-axis
    range_y=[10,70],           # set range of y-axis
    # category_orders={'Year':[2015,2016,2017,2018,2019]},    # set a specific ordering of values per column
)

# violinfig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000


violinfig.update_layout(yaxis = dict(tickfont=dict(size=12)),xaxis=dict(tickfont=dict(size=1)))

violinfig.add_trace(px.line(prcl_df, x="CIR_Conf",
                    y="percentil95").data[0])

pio.show(violinfig)

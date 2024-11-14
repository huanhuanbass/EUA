import streamlit as st
st.set_page_config(layout="wide")
import plotly.express as px

import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta
from dateutil import relativedelta
from pandas.tseries.offsets import BDay
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import yfinance as yf

import plotly.graph_objects as go

draft_template = go.layout.Template()
draft_template.layout.annotations = [
    dict(
        name="draft watermark",
        text="COFCO Internal Use Only",
        textangle=0,
        opacity=0.1,
        font=dict(color="black", size=70),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
    )
]


#Getting EUA Data
st.text('----Getting EUA Data...')

@st.cache_data(ttl='24h')
def load_eua_data():

    eua_dec=pd.read_csv('碳排放期货历史数据.csv')
    eua_dec.rename(columns={'日期':'Date','收盘':'Close','开盘':'Open','高':'High','低':'Low','交易量':'Volume','涨跌幅':'DoD'},inplace=True)

    return eua_dec

eua_dec=load_eua_data()


if 'eua_dec' not in st.session_state:
    st.session_state['eua_dec']=eua_dec


st.text('EUA Data Retrieved!')


def update_data():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.cache_data.clear()
st.button('Update Data',on_click=update_data)
st.text('Data is automatically reloaded for potential updates every 24 hours.')
st.text('If you would like to trigger the reload right now, please click on the above "Update Data" button.')


cutoff = pd.to_datetime('today')
curryear=cutoff.year

plot_ticks='inside'
plot_tickwidth=2
plot_ticklen=10
plot_title_font_color='dodgerblue'
plot_title_font_size=25
plot_legend_font_size=15
plot_axis=dict(tickfont = dict(size=15))

st.title('EUA')
st.text('Dry Bulk Freight (EUA) Interactive Dashboard')

st.markdown('## **Candle Chart for Energy Products**')


eua_dec=st.session_state['eua_dec']
eua_d=eua_dec.copy()



euayf= yf.Ticker("CO2.L")
eua=euayf.history(period="20y")
eua.reset_index(inplace=True)

eua_pt=eua[['Date','Close']]
eua_pt.set_index('Date',inplace=True)
eua_pt.rename(columns={'Close':'EUA Spot'},inplace=True)
eua_pt.index=pd.to_datetime(eua_pt.index)
eua_pt.index=eua_pt.index.tz_localize(None)

eua['Date']=pd.to_datetime(eua['Date'])
eua_w=eua.groupby(pd.Grouper(key='Date',freq='W')).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).reset_index()
eua_m=eua.groupby(pd.Grouper(key='Date',freq='M')).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).reset_index()


eua_d_pt=eua_d[['Date','Close']]
eua_d_pt.set_index('Date',inplace=True)
eua_d_pt.rename(columns={'Close':'EUA Dec Rolling'},inplace=True)
eua_d_pt.index=pd.to_datetime(eua_d_pt.index)
energy=pd.merge(eua_pt,eua_d_pt,left_index=True,right_index=True,how='outer')

eua_d['Date']=pd.to_datetime(eua_d['Date'])
eua_d_w=eua_d.groupby(pd.Grouper(key='Date',freq='W')).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).reset_index()
eua_d_m=eua_d.groupby(pd.Grouper(key='Date',freq='M')).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).reset_index()


ttfyf= yf.Ticker("TTF=F")
ttf=ttfyf.history(period="20y")
ttf.reset_index(inplace=True)

ttf_pt=ttf[['Date','Close']]
ttf_pt.set_index('Date',inplace=True)
ttf_pt.rename(columns={'Close':'TTF'},inplace=True)
ttf_pt.index=pd.to_datetime(ttf_pt.index)
ttf_pt.index=ttf_pt.index.tz_localize(None)
energy=pd.merge(energy,ttf_pt,left_index=True,right_index=True,how='outer')

ttf['Date']=pd.to_datetime(ttf['Date'])
ttf_w=ttf.groupby(pd.Grouper(key='Date',freq='W')).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).reset_index()
ttf_m=ttf.groupby(pd.Grouper(key='Date',freq='M')).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).reset_index()

apiyf= yf.Ticker("MTF=F")
api=apiyf.history(period="20y")
api.reset_index(inplace=True)

api_pt=api[['Date','Close']]
api_pt.set_index('Date',inplace=True)
api_pt.rename(columns={'Close':'API2'},inplace=True)
api_pt.index=pd.to_datetime(api_pt.index)
api_pt.index=api_pt.index.tz_localize(None)
energy=pd.merge(energy,api_pt,left_index=True,right_index=True,how='outer')

api['Date']=pd.to_datetime(api['Date'])
api_w=api.groupby(pd.Grouper(key='Date',freq='W')).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).reset_index()
api_m=api.groupby(pd.Grouper(key='Date',freq='M')).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).reset_index()

wtiyf= yf.Ticker("CL=F")
wti=wtiyf.history(period="20y")
wti.reset_index(inplace=True)

wti_pt=wti[['Date','Close']]
wti_pt.set_index('Date',inplace=True)
wti_pt.rename(columns={'Close':'WTI'},inplace=True)
wti_pt.index=pd.to_datetime(wti_pt.index)
wti_pt.index=wti_pt.index.tz_localize(None)
energy=pd.merge(energy,wti_pt,left_index=True,right_index=True,how='outer')

wti['Date']=pd.to_datetime(wti['Date'])
wti_w=wti.groupby(pd.Grouper(key='Date',freq='W')).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).reset_index()
wti_m=wti.groupby(pd.Grouper(key='Date',freq='M')).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).reset_index()


brentyf= yf.Ticker("BZ=F")
brent=brentyf.history(period="20y")
brent.reset_index(inplace=True)

brent_pt=brent[['Date','Close']]
brent_pt.set_index('Date',inplace=True)
brent_pt.rename(columns={'Close':'Brent'},inplace=True)
brent_pt.index=pd.to_datetime(brent_pt.index)
brent_pt.index=brent_pt.index.tz_localize(None)
energy=pd.merge(energy,brent_pt,left_index=True,right_index=True,how='outer')

brent['Date']=pd.to_datetime(brent['Date'])
brent_w=brent.groupby(pd.Grouper(key='Date',freq='W')).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).reset_index()
brent_m=brent.groupby(pd.Grouper(key='Date',freq='M')).agg({'Open':'first','High':'max','Low':'min','Close':'last'}).reset_index()


cdtype=st.selectbox('Select Product',options=['Carbon EUA Dec Rolling','Carbon EUA Spot','Natural Gas TTF','Coal API2','Crude Oil WTI','Crude Oil Brent'],key='cdtype')
yr1=st.number_input('Input Start Year',min_value=2005,max_value=curryear,value=curryear-1,step=1,key='yr1')
cdfreq=st.selectbox('Select Frequency',options=['Daily','Weekly','Monthly'],key='cdfreq')

if cdtype=='Carbon EUA Spot':
    if cdfreq=='Daily':
        eua_s_d=eua[eua['Date'].dt.year>=yr1]
        cddata=eua_s_d
    elif cdfreq=='Weekly':
        eua_s_w=eua_w[eua_w['Date'].dt.year>=yr1]
        cddata=eua_s_w
    elif cdfreq=='Monthly':
        eua_s_m=eua_m[eua_m['Date'].dt.year>=yr1]
        cddata=eua_s_m

if cdtype=='Carbon EUA Dec Rolling':
    if cdfreq=='Daily':
        eua_d_d=eua_d[eua_d['Date'].dt.year>=yr1]
        cddata=eua_d_d
    elif cdfreq=='Weekly':
        eua_d_w=eua_d_w[eua_d_w['Date'].dt.year>=yr1]
        cddata=eua_d_w
    elif cdfreq=='Monthly':
        eua_d_m=eua_d_m[eua_d_m['Date'].dt.year>=yr1]
        cddata=eua_d_m

if cdtype=='Natural Gas TTF':
    if cdfreq=='Daily':
        ttf_s_d=ttf[ttf['Date'].dt.year>=yr1]
        cddata=ttf_s_d
    elif cdfreq=='Weekly':
        ttf_s_w=ttf_w[ttf_w['Date'].dt.year>=yr1]
        cddata=ttf_s_w
    elif cdfreq=='Monthly':
        ttf_s_m=ttf_m[ttf_m['Date'].dt.year>=yr1]
        cddata=ttf_s_m

if cdtype=='Coal API2':
    if cdfreq=='Daily':
        api_s_d=api[api['Date'].dt.year>=yr1]
        cddata=api_s_d
    elif cdfreq=='Weekly':
        api_s_w=api_w[api_w['Date'].dt.year>=yr1]
        cddata=api_s_w
    elif cdfreq=='Monthly':
        api_s_m=api_m[api_m['Date'].dt.year>=yr1]
        cddata=api_s_m

if cdtype=='Crude Oil WTI':
    if cdfreq=='Daily':
        wti_s_d=wti[wti['Date'].dt.year>=yr1]
        cddata=wti_s_d
    elif cdfreq=='Weekly':
        wti_s_w=wti_w[wti_w['Date'].dt.year>=yr1]
        cddata=wti_s_w
    elif cdfreq=='Monthly':
        wti_s_m=wti_m[wti_m['Date'].dt.year>=yr1]
        cddata=wti_s_m

if cdtype=='Crude Oil Brent':
    if cdfreq=='Daily':
        brent_s_d=brent[brent['Date'].dt.year>=yr1]
        cddata=brent_s_d
    elif cdfreq=='Weekly':
        brent_s_w=brent_w[brent_w['Date'].dt.year>=yr1]
        cddata=brent_s_w
    elif cdfreq=='Monthly':
        brent_s_m=brent_m[brent_m['Date'].dt.year>=yr1]
        cddata=brent_s_m




candle=go.Figure(data=[go.Candlestick(x=cddata['Date'],open=cddata['Open'],high=cddata['High'],low=cddata['Low'],close=cddata['Close'])])
candle.update_layout(title=str(cdtype)+' '+cdfreq+' Candle Chart',width=1000,height=500)
candle.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
candle.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
candle.update_layout(xaxis_rangeslider_visible=False)
candle.update_layout(template=draft_template)
st.plotly_chart(candle)


st.markdown('## **Line Chart for Energy Products**')

rangelist00=st.selectbox('Select Range',options=['Last Year to Date','Year to Date','Month to Date','Last Week to Date','All'],key='rg00')
sllist00=st.multiselect('Select Products',options=energy.columns,default=['TTF','API2','EUA Dec Rolling'],key='sl00')
energy_sl=energy[sllist00]

today = pd.to_datetime('today')
if rangelist00=='Last Week to Date':
    rangestart00=today - timedelta(days=today.weekday()) + timedelta(days=6, weeks=-2)
elif rangelist00=='Month to Date':
    rangestart00=date(today.year,today.month,1)
elif rangelist00=='Year to Date':
    rangestart00=date(today.year,1,1)
elif rangelist00=='Last Year to Date':
    rangestart00=date(today.year-1,1,1)
else:
    rangestart00=date(2015,1,1)

energy_sl=energy_sl[pd.to_datetime(energy_sl.index)>=pd.to_datetime(rangestart00)]
lplot=px.line(energy_sl,width=1000,height=500,title='EUA and Related Products Historical Price')
lplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
lplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
lplot.update_layout(template=draft_template)
st.plotly_chart(lplot)




st.markdown('## **EUA Price**')


eua_pt1=eua_f.pivot_table(index='Date',columns='Fixed Contract',values='Amount',aggfunc='mean')
eua_pt1.index=pd.to_datetime(eua_pt1.index,dayfirst=True)
eua_pt1.sort_index(inplace=True)

eua_pt2=eua_f.pivot_table(index='Date',columns='Rolling Month Gap',values='Amount',aggfunc='mean')
eua_pt2.sort_index(inplace=True)
eua_pt2.columns=eua_pt2.columns.astype('int64')


eua.set_index('Date',inplace=True)
eua.sort_index(ascending=True,inplace=True)
eua_close=eua[['Close']]
eua_close.index=eua_close.index.tz_localize(None)
eua_close.rename(columns={'Close':'Spot'},inplace=True)



eua_d.set_index('Date',inplace=True)
eua_d.sort_index(ascending=True,inplace=True)
eua_d_close=eua_d[['Close']]
eua_d_close.index=eua_d_close.index.tz_localize(None)
eua_d_close.rename(columns={'Close':'Dec Rolling'},inplace=True)


tooday=eua_pt1.index.max()


eua_pt1=pd.merge(eua_d_close,eua_pt1,left_index=True,right_index=True,how='outer')
eua_pt2=pd.merge(eua_d_close,eua_pt2,left_index=True,right_index=True,how='outer')

eua_pt1=pd.merge(eua_close,eua_pt1,left_index=True,right_index=True,how='outer')
eua_pt2=pd.merge(eua_close,eua_pt2,left_index=True,right_index=True,how='outer')

tday=eua_pt1.index.max()
lday=tday-BDay(1)
l2day=tday-BDay(2)
l3day=tday-BDay(3)
l4day=tday-BDay(4)
lweek=tday-BDay(5)
l2week=tday-BDay(10)
l3week=tday-BDay(15)
lmonth=tday-BDay(20)
l2month=tday-BDay(45)

s0='Spot'

for k in range(30):
    exec(f'm{k}=str((tday + relativedelta.relativedelta(months=k)).year)+\'_M\'+str((tday + relativedelta.relativedelta(months=k)).month)')

for y in range(10):
    exec(f'decy{y}=str(tday.year+y)+\'_M12\'')
    




st.header('EUA Spot and Forward Contracts Line Chart')
st.markdown('#### **----Fixed Contracts**')
rangelist1=st.selectbox('Select Range',options=['Last Year to Date','Year to Date','Month to Date','Last Week to Date','All'],key='rg1')
sllist1=st.multiselect('Select Contracts',options=eua_pt1.columns,default=['Dec Rolling'],key='sl1')
eua_sl=eua_pt1[sllist1]

today = pd.to_datetime('today')
if rangelist1=='Last Week to Date':
    rangestart1=today - timedelta(days=today.weekday()) + timedelta(days=6, weeks=-2)
elif rangelist1=='Month to Date':
    rangestart1=date(today.year,today.month,1)
elif rangelist1=='Year to Date':
    rangestart1=date(today.year,1,1)
elif rangelist1=='Last Year to Date':
    rangestart1=date(today.year-1,1,1)
else:
    rangestart1=date(2015,1,1)

eua_sl=eua_sl[pd.to_datetime(eua_sl.index)>=pd.to_datetime(rangestart1)]
lplot=px.line(eua_sl,width=1000,height=500,title='EUA Spot and Fixed Forward Contract Historical Price')
lplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
lplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
lplot.update_layout(template=draft_template)
st.plotly_chart(lplot)




st.header('EUA Spot and Dec Rolling Contracts Seasonality')
contractlist_211=st.selectbox('Select Contract',options=['Dec Rolling','Spot'],key='211')
freq=st.radio('Select Frequency',options=['Weekly','Monthly','Quarterly'],key='spotfreq')
eua_sp=eua_pt2[[contractlist_211]]
eua_sp.index=pd.to_datetime(eua_sp.index)


if freq=='Weekly':
    eua_sp['Year']=eua_sp.index.year
    eua_sp['Week']=eua_sp.index.isocalendar().week
    eua_sp.loc[eua_sp[eua_sp.index.date==date(2016,1,2)].index,'Week']=0
    eua_sp.loc[eua_sp[eua_sp.index.date==date(2021,1,2)].index,'Week']=0
    eua_sp.loc[eua_sp[eua_sp.index.date==date(2022,1,1)].index,'Week']=0
    yrlist=list(eua_sp['Year'].unique())
    yrlist.sort(reverse=True)
    yrsl=st.multiselect('Select Years',options=yrlist,default=np.arange(curryear-3,curryear+1),key='spotyear1')
    eua_sp=eua_sp[eua_sp['Year'].isin(yrsl)]
    eua_sppt=eua_sp.pivot_table(index='Week',columns='Year',values=contractlist_211,aggfunc='mean')

    spotplot=px.line(eua_sppt,width=1000,height=500,title='EUA Spot Weekly Seasonality')
    spotplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
    spotplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
    spotplot['data'][-1]['line']['width']=5
    spotplot['data'][-1]['line']['color']='black'
    spotplot.update_layout(template=draft_template)
    st.plotly_chart(spotplot)

elif freq=='Monthly':
    eua_sp['Year']=eua_sp.index.year
    eua_sp['Month']=eua_sp.index.month
    yrlist=list(eua_sp['Year'].unique())
    yrlist.sort(reverse=True)
    yrsl=st.multiselect('Select Years',options=yrlist,default=np.arange(curryear-3,curryear+1),key='spotyear2')
    eua_sp=eua_sp[eua_sp['Year'].isin(yrsl)]
    eua_sppt=eua_sp.pivot_table(index='Month',columns='Year',values=contractlist_211,aggfunc='mean')

    spotplot=px.line(eua_sppt,width=1000,height=500,title='EUA Spot Monthly Seasonality')
    spotplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
    spotplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
    spotplot['data'][-1]['line']['width']=5
    spotplot['data'][-1]['line']['color']='black'
    spotplot.update_layout(template=draft_template)
    st.plotly_chart(spotplot)

elif freq=='Quarterly':
    eua_sp['Year']=eua_sp.index.year
    eua_sp['Quarter']=eua_sp.index.quarter
    yrlist=list(eua_sp['Year'].unique())
    yrlist.sort(reverse=True)
    yrsl=st.multiselect('Select Years',options=yrlist,default=np.arange(curryear-3,curryear+1),key='spotyear3')
    eua_sp=eua_sp[eua_sp['Year'].isin(yrsl)]
    eua_sppt=eua_sp.pivot_table(index='Quarter',columns='Year',values=contractlist_211,aggfunc='mean')

    spotplot=px.line(eua_sppt,width=1000,height=500,title='EUA Spot Quarterly Seasonality')
    spotplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
    spotplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
    spotplot['data'][-1]['line']['width']=5
    spotplot['data'][-1]['line']['color']='black'
    spotplot.update_layout(template=draft_template)
    st.plotly_chart(spotplot)


st.header('EUA Summary')
eua_df=eua_pt1[['Spot','Dec Rolling']]

eua_df.index=eua_df.index.date

eua_=pd.concat([eua_df.loc[[tooday.date()]],eua_df.loc[[lday.date()]],eua_df.loc[[lweek.date()]],eua_df.loc[[lmonth.date()]]])
st.write(eua_.style.format('{:,.2f}'))


st.markdown('#### **Change**')
eua_.loc['DoD Chg']=eua_df.loc[tooday.date()]-eua_df.loc[lday.date()]
eua_.loc['WoW Chg']=eua_df.loc[tooday.date()]-eua_df.loc[lweek.date()]
eua_.loc['MoM Chg']=eua_df.loc[tooday.date()]-eua_df.loc[lmonth.date()]
eua_chg=pd.concat([eua_.loc[['DoD Chg']],eua_.loc[['WoW Chg']],eua_.loc[['MoM Chg']]])
st.write(eua_chg.style.format('{:,.2f}'))

st.markdown('#### **Change in Percentage**')
eua_.loc['DoD Chg %']=eua_.loc['DoD Chg']/eua_df.loc[lday.date()]
eua_.loc['WoW Chg %']=eua_.loc['WoW Chg']/eua_df.loc[lweek.date()]
eua_.loc['MoM Chg %']=eua_.loc['MoM Chg']/eua_df.loc[lmonth.date()]
eua_chgpct=pd.concat([eua_.loc[['DoD Chg %']],eua_.loc[['WoW Chg %']],eua_.loc[['MoM Chg %']]])
st.write(eua_chgpct.style.format('{:,.2%}'))

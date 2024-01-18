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


#Getting EUA Data
st.text('----Getting EUA Data...')

@st.cache_data(ttl='24h')
def load_eua_data():
    eua=pd.read_csv('Historical data - EUA Price.csv')

    eua_s=eua[eua['Period']=='SPOT']
    eua_f=eua[eua['Period']!='SPOT']

    eua_f[['Month','Year']]=eua_f['Period'].str.split('-',expand=True)
    eua_f['Month'].replace({'JAN':'1','FEB':'2','MAR':'3','APR':'4','MAY':'5','JUN':'6',
                         'JUL':'7','AUG':'8','SEP':'9','OCT':'10','NOV':'11','DEC':'12'},inplace=True)

    eua_f['Fixed Contract']=eua_f['Year']+'_M'+eua_f['Month']
    eua_f['Month']=pd.to_numeric(eua_f['Month'])
    eua_f['Year']=pd.to_numeric(eua_f['Year'])
    eua_f['Archive Month']=pd.to_datetime(eua_f['Date']).dt.month
    eua_f['Archive Year']=pd.to_datetime(eua_f['Date']).dt.year
    eua_f['Rolling Month Gap']=(eua_f['Year']-eua_f['Archive Year'])*12+(eua_f['Month']-eua_f['Archive Month'])

    eua_s['Date']=pd.to_datetime(eua_s['Date'])
    eua_s['Month']=eua_s['Date'].dt.month
    eua_s['Year']=eua_s['Date'].dt.year
    eua_s['Fixed Contract']=eua_s['Year'].astype('str')+'_M'+eua_s['Month'].astype('str')
    eua_s['Archive Month']=pd.to_datetime(eua_s['Date']).dt.month
    eua_s['Archive Year']=pd.to_datetime(eua_s['Date']).dt.year
    eua_s['Rolling Month Gap']=(eua_s['Year']-eua_s['Archive Year'])*12+(eua_s['Month']-eua_s['Archive Month'])
    
    eua_f['Date']=pd.to_datetime(eua_f['Date'])

    eua_f=pd.concat([eua_s,eua_f])

    fxyf=yf.Ticker("EURUSD=X")
    fx=fxyf.history(period="20y")
    fx.index=fx.index.tz_localize(None)
    fx.reset_index(inplace=True)
    fxclose=fx[['Date','Close']]
    fxclose.rename(columns={'Close':'FX'},inplace=True)

    eua_f=pd.merge(eua_f,fxclose,left_on='Date',right_on='Date',how='left')
    eua_f.rename(columns={'Amount':'Amount in USD'},inplace=True)
    eua_f['Amount']=eua_f['Amount in USD']/eua_f['FX']

    eua_dec=pd.read_csv('European Union Allowance.csv')
    eua_dec.rename(columns={'日期':'Date','收盘':'Close','开盘':'Open','高':'High','低':'Low','交易量':'Volume','涨跌幅':'DoD'},inplace=True)

    return eua_f, eua_dec

eua_f,eua_dec=load_eua_data()


if 'eua_f' not in st.session_state:
    st.session_state['eua_f']=eua_f
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


eua_f=st.session_state['eua_f']
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


cdtype=st.selectbox('Select Product',options=['Carbon EUA Spot','Carbon EUA Dec Rolling','Natural Gas TTF','Coal API2','Crude Oil WTI','Crude Oil Brent'],key='cdtype')
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
st.plotly_chart(candle)


st.markdown('## **Line Chart for Energy Products**')

rangelist00=st.selectbox('Select Range',options=['Last Year to Date','Year to Date','Month to Date','Last Week to Date','All'],key='rg00')
sllist00=st.multiselect('Select Products',options=energy.columns,default=['EUA Spot','TTF','API2'],key='sl00')
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
lplot=px.line(energy_sl,width=1000,height=500,title='Energy Related Contracts Historical Price')
lplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
lplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
st.plotly_chart(lplot)




st.markdown('## **EUA Price**')

eua_f.sort_values(by='Date',ascending=True,inplace=True)

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
sllist1=st.multiselect('Select Contracts',options=eua_pt1.columns,default=['Spot',m0,m1,'Dec Rolling',decy0,decy1],key='sl1')
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
st.plotly_chart(lplot)



st.header('EUA Technical Analysis')
st.markdown('#### **----Fixed Contracts**')

rangelist0=st.selectbox('Select Range',options=['Last Year to Date','Year to Date','Month to Date','Last Week to Date','All'],key='rg0')
if rangelist0=='Last Week to Date':
    rangestart0=today - timedelta(days=today.weekday()) + timedelta(days=6, weeks=-2)
elif rangelist0=='Month to Date':
    rangestart0=date(today.year,today.month,1)
elif rangelist0=='Year to Date':
    rangestart0=date(today.year,1,1)
elif rangelist0=='Last Year to Date':
    rangestart0=date(today.year-1,1,1)
else:
    rangestart0=date(2015,1,1)

contractlist=st.selectbox('Select Spot or Forward Contract',options=list(eua_pt1.columns))
bb=st.number_input('Bollinger Bands Window',value=20)
ma1=st.number_input('Short Term Moving Average Window',value=20)
ma2=st.number_input('Long Term Moving Average Window',value=50)


eua_contract=eua_pt1[[contractlist]]
eua_contract.dropna(inplace=True)

eua_contract.sort_index(inplace=True)
indicator_mast = SMAIndicator(close=eua_contract[contractlist], window=ma1)
indicator_malt = SMAIndicator(close=eua_contract[contractlist], window=ma2)
indicator_bb = BollingerBands(close=eua_contract[contractlist], window=bb, window_dev=2)
eua_contract['ma_st'] = indicator_mast.sma_indicator()
eua_contract['ma_lt'] = indicator_malt.sma_indicator()
eua_contract['bb_m'] = indicator_bb.bollinger_mavg()
eua_contract['bb_h'] = indicator_bb.bollinger_hband()
eua_contract['bb_l'] = indicator_bb.bollinger_lband()

eua_contract=eua_contract[pd.to_datetime(eua_contract.index)>=pd.to_datetime(rangestart0)]
contractplot=px.line(eua_contract,width=1000,height=500,title='EUA '+contractlist+' Fixed Contract Bollinger Bands and Moving Average')
contractplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
contractplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
st.plotly_chart(contractplot)



st.header('EUA Spot and Dec Rolling Contracts Seasonality')
contractlist_211=st.selectbox('Select Contract',options=['Spot','Dec Rolling'],key='211')
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
    st.plotly_chart(spotplot)

st.header('EUA Forward Curve')
sllist2=st.multiselect('Select Contracts',options=eua_pt1.columns,default=['Spot',m0,m1,decy0,decy1,decy2,decy3,decy4,decy5],key='2')
eua_fc=eua_pt1[sllist2]
eua_fct=eua_fc.transpose()


tday=tooday.date()
lday=tday-BDay(1)
l2day=tday-BDay(2)
l3day=tday-BDay(3)
l4day=tday-BDay(4)
lweek=tday-BDay(5)
l2week=tday-BDay(10)
l3week=tday-BDay(15)
lmonth=tday-BDay(20)
l2month=tday-BDay(45)


lday=lday.date()
l2day=l2day.date()
l3day=l3day.date()
l4day=l4day.date()
lweek=lweek.date()
l2week=l2week.date()
l3week=l3week.date()
lmonth=lmonth.date()
l2month=l2month.date()


sllist3=st.multiselect('Select Dates',options=eua_fct.columns.date,default=[tday,lday,l2day,lweek,l2week,lmonth,l2month],key='3')
eua_fctsl=eua_fct[sllist3]
fctplot=px.line(eua_fctsl,width=1000,height=500,title='EUA Forward Curve')
fctplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
fctplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
st.plotly_chart(fctplot)

st.markdown('#### **----Implied Interest Rate**')

eua_pt1t=eua_pt1.transpose()

eua_tday=eua_fct[[tday]]
eua_tooday=eua_pt1t[[tday]]


ir1=st.selectbox('Select Contract 1',options=[m0]+list(eua_pt1.columns),key='ir1')
eua_ir1=eua_tooday.loc[ir1,:]
eua_ir1=eua_ir1.iloc[0]
ir1v=st.number_input('Input Price for Contract 1',value=eua_ir1)

ir2list=[decy0]+list(eua_pt1.columns)
ir2list.remove('Spot')
ir2=st.selectbox('Select Contract 2 (exclude SPOT)',options=ir2list,key='ir2')

eua_ir2=eua_tooday.loc[ir2,:]
eua_ir2=eua_ir2.iloc[0]
ir2v=st.number_input('Input Price for Contract 2',value=eua_ir2)

mg=st.number_input('Input Margin Estimation for Future Contracts',value=0.1)

if ir1=='Spot':
    yr_ir1,m_ir1=m0.split('_M')
    dgap=30-tday.day

else:
    yr_ir1,m_ir1=ir1.split('_M')
    dgap=0


yr_ir2,m_ir2=ir2.split('_M')

yr_ir0,m_ir0=m0.split('_M')

yr_ir1=int(yr_ir1)
m_ir1=int(m_ir1)
yr_ir2=int(yr_ir2)
m_ir2=int(m_ir2)
mgap=(yr_ir2-yr_ir1)*12+(m_ir2-m_ir1)+dgap/30

yr_ir0=int(yr_ir0)
m_ir0=int(m_ir0)

t1=(yr_ir1-yr_ir0)+(m_ir1-m_ir0)/12+(30-tday.day)/360
t2=(yr_ir2-yr_ir1)+(m_ir2-m_ir1)/12


if ir1=='Spot':
    rir=((ir2v-ir2v*mg)-(ir1v-ir2v*mg))/(ir1v-ir2v*mg)
    yrir=(1+rir)**(12/mgap)-1
else:
    yrir=(ir2v-ir1v)/(ir1v*(1-mg)*t2-(ir2v-ir1v)*mg*(t1+t2))


yrir="{:.2%}".format(yrir)


st.markdown('##### :blue[Implied Interest Rate Between] '+str(ir1)+' :blue[and] '+str(ir2)+' :blue[:] '+str(yrir))

st.header('EUA Interest Rate Arbitrage')
ourir=st.number_input('Input Our Financing Cost',value=0.06)
ourmg=st.number_input('Input Margin Estimation for Future Contracts',value=0.1,key='mg2')

st.markdown('#### **----From Spot to Future**')
eua_sp=eua_tday.loc['Spot',:]
eua_sp=eua_sp.iloc[0]
ourspot=st.number_input('Input Our Spot Price',value=eua_sp)
sp_dgap=30-tday.day
ourir_sp_fm=(1+ourir)**(sp_dgap/360)-1
eua_sp_fm=(ourspot*(1+ourir_sp_fm)/(1+ourmg*ourir_sp_fm))

yr_fm,m_fm=m0.split('_M')
yr_fd,m_fd=decy0.split('_M')
sp_mgap=(int(yr_fd)-int(yr_fm))*12+(int(m_fd)-int(m_fm))+sp_dgap/30
ourir_sp_fd=(1+ourir)**(sp_mgap/12)-1
eua_sp_fd=(ourspot*(1+ourir_sp_fd)/(1+ourmg*ourir_sp_fd))

gap_spm1=(sp_dgap)/30+1
ir_spm1=(1+ourir)**(gap_spm1/12)-1
eua_sp_m1=(ourspot*(1+ir_spm1)/(1+ourmg*ir_spm1))

gap_sp1=sp_mgap+12*1
ir_sp1=(1+ourir)**(gap_sp1/12)-1
eua_sp_d1=(ourspot*(1+ir_sp1)/(1+ourmg*ir_sp1))

gap_sp2=sp_mgap+12*2
ir_sp2=(1+ourir)**(gap_sp2/12)-1
eua_sp_d2=(ourspot*(1+ir_sp2)/(1+ourmg*ir_sp2))

gap_sp3=sp_mgap+12*3
ir_sp3=(1+ourir)**(gap_sp3/12)-1
eua_sp_d3=(ourspot*(1+ir_sp3)/(1+ourmg*ir_sp3))

gap_sp4=sp_mgap+12*4
ir_sp4=(1+ourir)**(gap_sp4/12)-1
eua_sp_d4=(ourspot*(1+ir_sp4)/(1+ourmg*ir_sp4))

gap_sp5=sp_mgap+12*5
ir_sp5=(1+ourir)**(gap_sp5/12)-1
eua_sp_d5=(ourspot*(1+ir_sp5)/(1+ourmg*ir_sp5))

d1={'Implied Curve':pd.Series([ourspot,eua_sp_fm,eua_sp_m1,eua_sp_fd,eua_sp_d1,eua_sp_d2,eua_sp_d3,eua_sp_d4,eua_sp_d5],index=['Spot',m0,m1,decy0,decy1,decy2,decy3,decy4,decy5])}
df1=pd.DataFrame(d1)

st.write(df1.transpose())


st.markdown('#### **----From Front Month to Spot and Future**')
eua_fm=eua_tday.loc[m0,:]
eua_fm=eua_fm.iloc[0]
ourfm=st.number_input('Input Our Front Month Price',value=eua_fm)

fm_dgap=30-tday.day
ourir_fm_sp=(1+ourir)**(fm_dgap/360)-1
eua_fm_sp=ourfm*(1+ourmg*ourir_fm_sp)/(1+ourir_fm_sp)

yr_fm,m_fm=m0.split('_M')
yr_fd,m_fd=decy0.split('_M')
fm_mgap=(int(yr_fd)-int(yr_fm))*12+(int(m_fd)-int(m_fm))+fm_dgap/30

ourir_fm_fd=(1+ourir)**(fm_mgap/12)-1
eua_fm_fd=(eua_fm_sp*(1+ourir_fm_fd)/(1+ourmg*ourir_fm_fd))

gap_fmm1=(fm_dgap)/30+1
ir_fmm1=(1+ourir)**(gap_fmm1/12)-1
eua_fm_m1=(eua_fm_sp*(1+ir_fmm1)/(1+ourmg*ir_fmm1))

gap_fm1=fm_mgap+12*1
ir_fm1=(1+ourir)**(gap_fm1/12)-1
eua_fm_d1=(eua_fm_sp*(1+ir_fm1)/(1+ourmg*ir_fm1))

gap_fm2=fm_mgap+12*2
ir_fm2=(1+ourir)**(gap_fm2/12)-1
eua_fm_d2=(eua_fm_sp*(1+ir_fm2)/(1+ourmg*ir_fm2))

gap_fm3=fm_mgap+12*3
ir_fm3=(1+ourir)**(gap_fm3/12)-1
eua_fm_d3=(eua_fm_sp*(1+ir_fm3)/(1+ourmg*ir_fm3))

gap_fm4=fm_mgap+12*4
ir_fm4=(1+ourir)**(gap_fm4/12)-1
eua_fm_d4=(eua_fm_sp*(1+ir_fm4)/(1+ourmg*ir_fm4))

gap_fm5=fm_mgap+12*5
ir_fm5=(1+ourir)**(gap_fm5/12)-1
eua_fm_d5=(eua_fm_sp*(1+ir_fm5)/(1+ourmg*ir_fm5))


d2={'Implied Curve':pd.Series([eua_fm_sp,ourfm,eua_fm_m1,eua_fm_fd,eua_fm_d1,eua_fm_d2,eua_fm_d3,eua_fm_d4,eua_fm_d5],index=['Spot',m0,m1,decy0,decy1,decy2,decy3,decy4,decy5])}
df2=pd.DataFrame(d2)
st.write(df2.transpose())

st.markdown('#### **----From Front Dec to Spot and Future**')
eua_fd=eua_tday.loc[decy0,:]
eua_fd=eua_fd.iloc[0]
ourfd=st.number_input('Input Our Front Dec Price',value=eua_fd)

fd_dgap=30-tday.day
yr_fm,m_fm=m0.split('_M')
yr_fd,m_fd=decy0.split('_M')
fd_mgap=(int(yr_fd)-int(yr_fm))*12+(int(m_fd)-int(m_fm))+fd_dgap/30

ourir_fd_sp=(1+ourir)**(fd_mgap/12)-1
eua_fd_sp=ourfd*(1+ourmg*ourir_fd_sp)/(1+ourir_fd_sp)

ourir_fd_fm=(1+ourir)**(fd_dgap/360)-1
eua_fd_fm=(eua_fd_sp*(1+ourir_fd_fm)/(1+ourmg*ourir_fd_fm))

gap_fdm1=(fd_dgap)/30+1
ir_fdm1=(1+ourir)**(gap_fdm1/12)-1
eua_fd_m1=(eua_fd_sp*(1+ir_fdm1)/(1+ourmg*ir_fdm1))

gap_fd1=fd_mgap+12*1
ir_fd1=(1+ourir)**(gap_fd1/12)-1
eua_fd_d1=(eua_fd_sp*(1+ir_fd1)/(1+ourmg*ir_fd1))

gap_fd2=fd_mgap+12*2
ir_fd2=(1+ourir)**(gap_fd2/12)-1
eua_fd_d2=(eua_fd_sp*(1+ir_fd2)/(1+ourmg*ir_fd2))

gap_fd3=fd_mgap+12*3
ir_fd3=(1+ourir)**(gap_fd3/12)-1
eua_fd_d3=(eua_fd_sp*(1+ir_fd3)/(1+ourmg*ir_fd3))

gap_fd4=fd_mgap+12*4
ir_fd4=(1+ourir)**(gap_fd4/12)-1
eua_fd_d4=(eua_fd_sp*(1+ir_fd4)/(1+ourmg*ir_fd4))

gap_fd5=fd_mgap+12*5
ir_fd5=(1+ourir)**(gap_fd5/12)-1
eua_fd_d5=(eua_fd_sp*(1+ir_fd5)/(1+ourmg*ir_fd5))

d3={'Implied Curve':pd.Series([eua_fd_sp,eua_fd_fm,eua_fd_m1,ourfd,eua_fd_d1,eua_fd_d2,eua_fd_d3,eua_fd_d4,eua_fd_d5],index=['Spot',m0,m1,decy0,decy1,decy2,decy3,decy4,decy5])}
df3=pd.DataFrame(d3)
st.write(df3.transpose())


st.header('EUA Time Spread')
st.markdown('#### **----Fixed Contracts**')
tsp1=st.selectbox('Select Contract 1',options=[m0]+list(eua_pt1.columns))
tsp2=st.selectbox('Select Contract 2',options=[decy0]+list(eua_pt1.columns))

if tsp1!=tsp2:
    eua_tsp=eua_pt1[[tsp1,tsp2]]
    eua_tsp.dropna(inplace=True)
    eua_tsp['Spread']=eua_tsp[tsp1]-eua_tsp[tsp2]
    tspplot=px.line(eua_tsp[['Spread']],width=1000,height=500,title='EUA Fixed Contract Time Spread: '+str(tsp1)+' minus '+str(tsp2))
    tspplot.update_xaxes(ticks=plot_ticks, tickwidth=plot_tickwidth,  ticklen=plot_ticklen)
    tspplot.update_layout(title_font_color=plot_title_font_color,title_font_size=plot_title_font_size,legend_font_size=plot_legend_font_size,xaxis=plot_axis,yaxis=plot_axis)
    st.plotly_chart(tspplot)





lday=tooday-BDay(1)
l2day=tooday-BDay(2)
l3day=tooday-BDay(3)
l4day=tooday-BDay(4)
lweek=tooday-BDay(5)
l2week=tooday-BDay(10)
l3week=tooday-BDay(15)
lmonth=tooday-BDay(20)
l2month=tooday-BDay(45)


st.header('EUA Summary')
eua_df=eua_pt1[['Spot',m0,m1,'Dec Rolling',decy0,decy1,decy2,decy3,decy4,decy5]]

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

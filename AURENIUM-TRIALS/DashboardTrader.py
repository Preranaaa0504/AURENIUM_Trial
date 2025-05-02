import requests, pandas as pd, streamlit as st
from API_KEY import api_key

st.header('Interactive Financial Dashboard')

endpoint = st.sidebar.selectbox('Endpoint: ',options=['convert','live_currencies_list','live_crypto_list','live',
                                            'historical','minute_historical','timeseries'])
url = f'https://marketdata.tradermade.com/api/v1/{endpoint}?api_key={api_key}'

if endpoint == 'convert':
    curr1 = st.sidebar.text_input('From Currency','USD')
    curr2 = st.sidebar.text_input('To Currency','EUR')
    amount = st.sidebar.text_input('Amount','1000')
    extension = f'&from={curr1}&to={curr2}&amount={amount}'
    url = url + extension
    data = requests.get(url).json()
    df = pd.DataFrame(data, index=['index']).T
    st.write(df)
elif endpoint == 'live_currencies_list':
    data = requests.get(url).json()
    df = pd.DataFrame(data['available_currencies'], index = ['index']).T
    st.write(df)
else:
    currency = st.sidebar.text_input('Forex','USDEUR')
    extension = f'&currency={currency}'
    if endpoint == 'historical':
        date1 = st.sidebar.date_input('Date')
        extension2 = f'&date={date1}'
        extension = extension+extension2
    if endpoint == 'minute_historical':
        date1 = st.sidebar.date_input('Date')
        time1 = st.sidebar.text_input('Time','11:00')
        date_time = str(date1)+'-'+str(time1)
        extension2 = f'&date_time={date_time}'
        extension = extension+extension2
    if endpoint == 'timeseries':
        start_date = st.sidebar.date_input('Start Date')
        end_date = st.sidebar.date_input('End Date')
        interval = st.sidebar.selectbox('Interval',options=['daily','hourly','minute'])
        extension2 = f'&start_date={start_date}&end_date={end_date}&interval={interval}'
        extension = extension+extension2
    url = url+extension
    data = requests.get(url).json()
    try:
        df = pd.DataFrame(data['quotes']).T
    except:
        df = pd.DataFrame(data, index=['index']).T
    st.write(df)
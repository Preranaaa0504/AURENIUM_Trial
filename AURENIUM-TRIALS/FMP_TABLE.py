import streamlit as st
import requests
import pandas as pd
import json


# Constants
BASE_URL = 'https://financialmodelingprep.com/api'
API_VERSION = 'v3'
API_KEY = 'KMnGVJA9NCTKL5SVpY4BWL54EcGXNiQ1'

st.header('Financial Modeling Prep Stock Screener')
symbol = st.sidebar.text_input('Ticker:', value='AAPL')

financial_data = st.sidebar.selectbox('Financial Data Type', options = ('income-statement', 'balance-sheet-statement', 'cash-flow-statement',
                                                                      'income-statement-growth', 'balance-sheet-statement-growth', 
                                                                      'cash-flow-statement-growth', 'ratios-ttm', 'ratios', 'financial-growth',
                                                                      'quote','rating', 'enterprise-values','key-metrics-ttm', 'key-metrics',
                                                                      'historical-rating', 'discounted-cash-flow','historical-discounted-cash-flow-statement',
                                                                      'historical-price-full', 'Historical Price smaller intervals'))

if financial_data == 'Historical Price smaller intervals':
    interval = st.sidebar.selectbox('Interval', options=('1min','5min', '15min', '30min','1hour', '4hour'))
    financial_data = 'historical-chart/'+interval

transpose = st.sidebar.selectbox('Transpose', options=('Yes', 'No'))

# Fix: Include API version in the URL
url = f'{BASE_URL}/{API_VERSION}/{financial_data}/{symbol}?apikey={API_KEY}'

try:
    response = requests.get(url)
    
    # Debug information
    st.sidebar.write(f"Status code: {response.status_code}")
    
    # Check if response is successful
    if response.status_code == 200:
        try:
            data = response.json()
            
            # Check if data is empty or null
            if not data:
                st.error(f"No data returned for {symbol}")
            else:
                if transpose == 'Yes':
                    df = pd.DataFrame(data).T
                else:
                    df = pd.DataFrame(data)
                st.write(df)
        except json.JSONDecodeError:
            st.error(f"Error parsing JSON response. Raw response: {response.text[:100]}...")
    else:
        st.error(f"API request failed with status code: {response.status_code}")
        st.write(f"Response: {response.text}")
        
except Exception as e:
    st.error(f"Error making API request: {str(e)}")
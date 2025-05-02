import streamlit as st, pandas as pd
from nselib import capital_market
from nselib import derivatives
import requests
from bs4 import BeautifulSoup


st.header('Indian Stock Dashboard')

data_type = st.selectbox('Type of Data: ', options=['Google Finance data','NSE Lib Data'])

if data_type == 'Google Finance data':

    ticker = st.sidebar.text_input('Symbol Code','INFY')
    exchange = st.sidebar.text_input('Exchange','NSE')

    url = f'https://www.google.com/finance/quote/{ticker}:{exchange}'

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    price = float(soup.find(class_='YMlKec fxKbKc').text.strip()[1:].replace(",",""))
    previous_close = float(soup.find(class_='P6K39c').text.strip()[1:].replace(",",""))
    revenue = soup.find(class_='QXDnM').text
    news = soup.find(class_='Yfwt5').text
    about = soup.find(class_='bLLb2d').text

    dict1 = {'Price':price,
            'Previous Price': previous_close,
            'Revenue':revenue,
            'News': news,
            'About':about}

    df = pd.DataFrame(dict1, index=['Extracted Data']).T

    st.write(df)



if data_type == 'NSE Lib Data':

    instrument = st.sidebar.selectbox('Instrument Type', options=('NSE Equity Market','NSE Derivatives Market'))
    if instrument == 'NSE Equity Market':
        data_info = st.sidebar.selectbox('Data to extract', options=('bhav_copy_equities','bhav_copy_with_delivery','equity_list','fno_equity_list',
                                                                    'market_watch_all_indices','nifty50_equity_list','block_deals_data','bulk_deal_data',
                                                                    'india_vix_data','short_selling_data','deliverable_position_data',
                                                                    'index_data','price_volume_and_deliverable_position_data','price_volume_data'))
        if (data_info =='equity_list') or (data_info =='fno_equity_list') or (data_info =='market_watch_all_indices') or (data_info =='nifty50_equity_list'):
            data = getattr(capital_market, data_info)()
        if (data_info =='bhav_copy_equities') or (data_info =='bhav_copy_with_delivery'):
            date = st.sidebar.text_input('Date','22-12-2023')
            data = getattr(capital_market, data_info)(date)
        if (data_info =='block_deals_data') or (data_info =='bulk_deal_data') or (data_info =='india_vix_data') or (data_info =='short_selling_data'):
            period_ = st.sidebar.text_input('Period','1M')
            data = getattr(capital_market, data_info)(period = period_)

    if instrument == 'NSE Derivatives Market':
        data_info = st.sidebar.selectbox('Data to extract', options=('expiry_dates_future', 'expiry_dates_option_index', 'fii_derivatives_statistics',
                                                                    'fno_bhav_copy','future_price_volume_data','nse_live_option_chain',
                                                                    'option_price_volume_data','participant_wise_open_interest',
                                                                    'participant_wise_trading_volume'))    
        if (data_info =='expiry_dates_future') or (data_info =='expiry_dates_option_index'):
            data = getattr(derivatives, data_info)()
        if (data_info =='fii_derivatives_statistics') or (data_info =='fno_bhav_copy') or (data_info =='participant_wise_open_interest') or (data_info =='participant_wise_trading_volume'):
            date = st.sidebar.text_input('Date','22-12-2023')
            data = getattr(derivatives, data_info)(date)
        if (data_info =='future_price_volume_data'):
            ticker = st.sidebar.text_input('Ticker','SBIN')
            type_ = st.sidebar.text_input('Instrument Type','FUTSTK')
            period_ = st.sidebar.text_input('Period','1M')
            data = derivatives.future_price_volume_data(ticker,type_,period=period_)
        if (data_info =='option_price_volume_data'):
            ticker = st.sidebar.text_input('Ticker','BANKNIFTY')
            type_ = st.sidebar.text_input('Instrument Type','OPTIDX')
            period_ = st.sidebar.text_input('Period','1M')
            data = derivatives.option_price_volume_data(ticker,type_,period=period_)
        if (data_info =='nse_live_option_chain'):
            ticker = st.sidebar.text_input('Ticker','BANKNIFTY')
            expiry_date = st.sidebar.text_input('Expiry Date','28-12-2023')
            data = derivatives.nse_live_option_chain(ticker,expiry_date=expiry_date)

    st.write(data)
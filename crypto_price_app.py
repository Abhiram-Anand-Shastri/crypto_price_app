import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from bs4 import BeautifulSoup
import requests
import json
from pycoingecko import CoinGeckoAPI  # <-- Import new library
import datetime as dt

# Set Page to expand to full width
st.set_page_config(layout="wide")

# --- Image (Added error handling) ---
try:
    image = Image.open('logo.png')
    st.image(image, width=500)
except FileNotFoundError:
    st.warning("logo.png file not found. Please make sure it's in the same directory.")

# Title
st.title('Crypto Web App')
st.markdown("""
This app retrieves prices along with other information regarding different cryptocurrencies from **CoinMarketCap**!
""")

# About
expander_bar = st.expander('About')
expander_bar.markdown("""
* **Made By:** <Your Name>
* **Data source:** [CoinMarketCap](http://coinmarketcap.com) & [CoinGecko](https://www.coingecko.com/en/api).
* **Credit:** Web scraper adapted from the Medium article *[Web Scraping Crypto Prices With Python](https://towardsdatascience.com/web-scraping-crypto-prices-with-python-41072ea5b5bf)* written by [Bryan Feng](https://medium.com/@bryanf).
""")

# Divide Page into columns
col1 = st.sidebar
col2, col3 = st.columns((1, 1))

col1.header('Input Options')

## Sidebar - Select currency price unit
currency_price_unit = col1.selectbox('Select currency for price',
                                     ('USD', 'BTC'))

#---------------------------------#

@st.cache_data(ttl=600)  # Cache data for 10 minutes
def load_data():
    """
    Scrapes live data from CoinMarketCap.
    """
    try:
        cmc = requests.get('https://coinmarketcap.com')
        cmc.raise_for_status() # Raise error for bad responses
        soup = BeautifulSoup(cmc.content, 'html.parser')

        data = soup.find('script', id='__NEXT_DATA__', type='application/json')
        coin_data = json.loads(data.contents[0])

        # --- Global Metrics ---
        global_metrics = coin_data['props']['pageProps']['globalMetrics']
        total_marketcap = global_metrics['marketCap']
        btc_market_share = global_metrics['btcDominance']
        eth_market_share = global_metrics['ethDominance']

        # --- Coin Data ---
        queries = coin_data['props']['dehydratedState']['queries']
        homepage_data_query = next(q for q in queries if q['queryKey'] == ['homepage-data', 1, 100])
        listings = homepage_data_query['state']['data']['data']['listing']['cryptoCurrencyList']

    except (requests.exceptions.RequestException, KeyError, StopIteration, TypeError, json.JSONDecodeError) as e:
        st.error(f"Failed to load live data from CoinMarketCap: {e}")
        return pd.DataFrame(), 0, 0, 0 # Return empty defaults

    coin_name = []
    coin_symbol = []
    market_cap = []
    percent_change_1h = []
    percent_change_24h = []
    percent_change_7d = []
    price = []
    volume_24h = []

    for i in listings:
        # Use 'slug' for coin_name as it's the ID for CoinGecko
        coin_name.append(i['slug'])
        coin_symbol.append(i['symbol'])

        try:
            quote_data = next(q for q in i['quotes'] if q['name'] == currency_price_unit)
        except StopIteration:
            # Handle if a coin doesn't have a quote for the selected currency
            quote_data = {} # Empty dict to avoid errors with .get()

        price.append(quote_data.get('price'))
        percent_change_1h.append(quote_data.get('percentChange1h'))
        percent_change_24h.append(quote_data.get('percentChange24h'))
        percent_change_7d.append(quote_data.get('percentChange7d'))
        market_cap.append(quote_data.get('marketCap'))
        volume_24h.append(quote_data.get('volume24h'))

    df = pd.DataFrame(columns=['coin_name', 'coin_symbol', 'market_cap', 'percent_change_1h', 'percent_change_24h', 'percent_change_7d', 'price', 'volume_24h'])
    df['coin_name'] = coin_name
    df['coin_symbol'] = coin_symbol
    df['price'] = price
    df['percent_change_1h'] = percent_change_1h
    df['percent_change_24h'] = percent_change_24h
    df['percent_change_7d'] = percent_change_7d
    df['market_cap'] = market_cap
    df['volume_24h'] = volume_24h

    # Drop rows where essential data might be missing
    df = df.dropna(subset=['price', 'market_cap'])
    df = df.astype({'market_cap': 'float64', 
                    'percent_change_1h': 'float64', 
                    'percent_change_24h': 'float64', 
                    'percent_change_7d': 'float64',
                    'price': 'float64',
                    'volume_24h': 'float64'})

    return df, total_marketcap, btc_market_share, eth_market_share

#---------------------------------#

df, total_marketcap, btc_market_share, eth_market_share = load_data()

#---------------------------------#

# Sidebar - Select cryptocurrencies
if not df.empty:
    sorted_coin = sorted(df['coin_symbol'])
    selected_coin = col1.multiselect('Cryptocurrency', sorted_coin,
                                     ['BTC', 'ETH', 'ADA', 'DOGE', 'BNB'])
else:
    col1.warning("Could not load cryptocurrency list.")
    selected_coin = [] # Ensure selected_coin is an empty list

# Filtering data
if not df.empty:
    selected_coin_df = df[(df['coin_symbol'].isin(selected_coin))]
else:
    selected_coin_df = pd.DataFrame() # Ensure selected_coin_df is an empty DataFrame

# Sidebar - Select Percent change timeframe
percent_timeframe = col1.selectbox('Percent change time frame',
                                     ['7d','24h', '1h'])

#---------------------------------#

percent_dict = {"7d": 'percent_change_7d',
                "24h": 'percent_change_24h',
                "1h": 'percent_change_1h'}
selected_percent_timeframe = percent_dict[percent_timeframe]

# Preparing data for plotting
if not df.empty:
    top_5_positive_change = df.nlargest(5, selected_percent_timeframe)
    top_5_negative_change = df.nsmallest(5, selected_percent_timeframe)
else:
    top_5_positive_change = pd.DataFrame()
    top_5_negative_change = pd.DataFrame()

if not selected_coin_df.empty:
    positive_change_selected_coins = \
        selected_coin_df[selected_coin_df[selected_percent_timeframe] > 0]
    negative_change_selected_coins = \
        selected_coin_df[selected_coin_df[selected_percent_timeframe] < 0]
else:
    positive_change_selected_coins = pd.DataFrame()
    negative_change_selected_coins = pd.DataFrame()

bar_chart_df = pd.concat([top_5_positive_change,
                          positive_change_selected_coins,
                          top_5_negative_change,
                          negative_change_selected_coins], axis=0)

if not bar_chart_df.empty:
    bar_chart_df['positive_percent_change'] = \
        bar_chart_df[selected_percent_timeframe] > 0
else:
    # Add empty column to prevent error if df is empty
    bar_chart_df['positive_percent_change'] = [] 

# Heading for Horizontal Bar Chart
col2.subheader(f'Bar plot of % Price Change')
col2.write(f'*Last {percent_timeframe} period*')

# Plotting Horizontal Bar Chart
plt.style.use('seaborn-v0_8-whitegrid') # <-- Corrected style name

fig, ax = plt.subplots()

if not bar_chart_df.empty:
    ax.barh(bar_chart_df['coin_symbol'],
            bar_chart_df[selected_percent_timeframe],
            color=bar_chart_df.positive_percent_change
            .map({True: 'lightblue', False: 'pink'}))

ax.set_xlabel('Percent Change', fontsize=17, labelpad=15)
ax.tick_params(axis='both', labelsize=13)
fig.tight_layout()
col2.pyplot(fig)

#---------------------------------#

def get_unit(max_market_cap):
    unit = 'less than ten million'
    # Ensure max_market_cap is a valid number before converting
    if pd.isna(max_market_cap) or max_market_cap == 0:
        return unit
    
    number_of_digits = len(str(int(max_market_cap)))

    if number_of_digits == 8:
        unit = 'tens of millions'
    elif number_of_digits == 9:
        unit = 'hundreds of millions'
    elif number_of_digits == 10:
        unit = 'billions'
    elif number_of_digits == 11:
        unit = 'tens of billions'
    elif number_of_digits == 12:
        unit = 'hundreds of billions'
    
    return unit

# Heading for Bar Chart
col3.subheader(f'Bar plot of Market Cap (Selected Cryptos)')
col3.write(f'*Last {percent_timeframe} period*')

# Plotting Bar Chart
fig, ax = plt.subplots()
if not selected_coin_df.empty:
    ax.bar(selected_coin_df['coin_symbol'],
           selected_coin_df['market_cap'])
ax.tick_params(axis='both', labelsize=15)

# Increasing size of exponenet
exponent = ax.yaxis.get_offset_text()
exponent.set_size(16)

# --- Robust error handling for empty dataframe ---
if not selected_coin_df.empty:
    max_market_cap = selected_coin_df['market_cap'].max()
    unit = get_unit(max_market_cap)
    if unit == 'less than ten million':
        ax.set_ylabel(f'Market Cap', fontsize=15, labelpad=15)
    else:
        ax.set_ylabel(f'Market Cap ({unit})', fontsize=15, labelpad=15)
else:
    ax.set_ylabel(f'Market Cap', fontsize=15, labelpad=15)
# --- End of fix ---

fig.tight_layout()
col3.pyplot(fig)

#---------------------------------#

col2.markdown("""
_________________________
""")

# Heading for Pie Chart
col2.header('**Market Share of Cryptos**')

# Preparing data for plotting
if total_marketcap > 0: # Avoid division by zero if API fails
    alt_coins_market_share = 100 - (btc_market_share + eth_market_share)
    percentages = [btc_market_share, eth_market_share, alt_coins_market_share]
    labels = ['Bitcoin', 'Ethereum', 'Alt Coins']

    # Plot Pie Chart
    fig, ax = plt.subplots()
    colors = ['#80dfff', 'pink', '#ffe699']
    ax.pie(percentages, labels=labels, colors=colors, autopct='%.1f%%')
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1), fontsize=10)

    # Display figure
    col2.pyplot(fig)
else:
    col2.warning("Could not calculate market share.")

#---------------------------------#
# --- START: NEW COINGECKO HISTORICAL DATA SECTION ---
#---------------------------------#

col3.markdown("""
_________________________
""")

# Getting Historical Time Series Data
# Using CoinGecko API - much more stable than scraping
cg = CoinGeckoAPI()

@st.cache_data(max_entries=50, ttl=86400) # Cache for 1 day
def get_historical_time_series_data(coin_id, vs_currency='usd', days=30):
    """
    Gets historical data from CoinGecko API.
    Note: coin_id is the API ID (e.g., 'bitcoin'), not the symbol (e.g., 'BTC').
    """
    try:
        # Get historical data
        history = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)
        
        # 'prices' is a list of [timestamp, price]
        prices = history['prices']
        
        # Convert to DataFrame
        historical_timeseries_df = pd.DataFrame(prices, columns=['Date', 'Close'])
        
        # Convert timestamp to datetime
        historical_timeseries_df['Date'] = pd.to_datetime(historical_timeseries_df['Date'], unit='ms')
        
        return historical_timeseries_df
    except Exception as e:
        # If CoinGecko fails
        print(f"CoinGecko API failed: {e}")
        return None

# Select crypto
if not df.empty:
    selected_crypto_symbol = col3.selectbox('Select crypto', (df['coin_symbol']))
else:
    selected_crypto_symbol = None # No crypto to select

# Heading for Line Graph
if selected_crypto_symbol:
    # --- NEW: Get the coin ID from the symbol ---
    # The API needs the id (e.g. 'bitcoin') not the symbol (e.g. 'BTC')
    try:
        # 'coin_name' column now holds the API ID (slug)
        selected_coin_id = df[df['coin_symbol'] == selected_crypto_symbol]['coin_name'].iloc[0]
    except (IndexError, KeyError):
        col3.error(f"Could not find API ID for symbol {selected_crypto_symbol}")
        selected_coin_id = None # Set to None to skip plotting

    if selected_coin_id:
        col3.header(f'{selected_crypto_symbol} over the last 30 days')
        
        # Call the new function with the coin ID
        historical_timeseries_df =\
            get_historical_time_series_data(coin_id=selected_coin_id, vs_currency='usd', days=30)

        # --- Error handling for CoinGecko ---
        if historical_timeseries_df is None or historical_timeseries_df.empty:
            col3.error(f"Error: Could not load historical data for {selected_crypto_symbol}.")
            col3.warning("The CoinGecko API may be down or this coin is not supported.")
        else:
            # Plot Line Graph
            fig, ax = plt.subplots()
            ax.plot(historical_timeseries_df['Date'],
                   historical_timeseries_df['Close'], color='green')

            ax.set_xlabel('Date', fontsize=15, labelpad=13)
            ax.set_ylabel('Closing Price ($)', fontsize=15, labelpad=15)
            ax.tick_params(axis='x', rotation=45)

            # Display figure
            col3.pyplot(fig)
else:
    col3.warning("Select a cryptocurrency to see its history.")

#---------------------------------#
# --- END: NEW COINGECKO SECTION ---
#---------------------------------#

col2.markdown("""
_________________________
""")

col2.header('**Tables**')

# Price Data Table

# Select columns
columns = ['coin_name', 'coin_symbol', 'market_cap', 'price', 'volume_24h']

# Check if columns exist before trying to select them
if not selected_coin_df.empty:
    # Filter only columns that actually exist in the DataFrame
    existing_columns = [col for col in columns if col in selected_coin_df.columns]
    selected_coin_price_info_df = selected_coin_df[existing_columns]
else:
    selected_coin_price_info_df = pd.DataFrame(columns=columns)


col2.subheader('Price Data of Selected Cryptocurrencies')
col2.write(selected_coin_price_info_df)

# Download CSV data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="crypto.csv">Download CSV File</a>'
    return href

if not selected_coin_price_info_df.empty:
    col2.markdown(filedownload(selected_coin_price_info_df),
                  unsafe_allow_html=True)

# Table of Percentage Change

# Drop columns not related to Percentage Change
if not selected_coin_df.empty:
    # Columns to drop
    cols_to_drop = ['market_cap', 'price', 'volume_24h']
    # Filter only columns that actually exist before trying to drop them
    existing_cols_to_drop = [col for col in cols_to_drop if col in selected_coin_df.columns]
    
    selected_coin_percent_change_df =\
        selected_coin_df.drop(columns=existing_cols_to_drop)

    col2.subheader('Percent Change Data of Select Cryptocurrencies')
    col2.write(selected_coin_percent_change_df)
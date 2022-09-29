from dash import Dash, Input, Output, callback, dash_table
import pandas as pd
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import requests
import json
import pandas as pd
import numpy as np
import yahooquery
from datetime import datetime
from yahooquery import Ticker
import yfinance as yf
import time
from dateutil.relativedelta import relativedelta
import pandas as pd

# start buildingDF function code
# -----------------------------------------------------------------------------------------------------------------------------------

#test tickers
ticker = 'AAPL'
tickerList = ['MSFT', 'GM']

daysOut_start = '30d'
daysOut_end = '120d'

# trading view filters 
mktCapMin = 5000000000
div_yield_recent = 2
StochK = 25
StochD = 25
macd_macd = 0
macd_signal = 0

#times
global tradingViewTime
global getCurrentPriceTime
global getFairValueTime
global getOptionsTime
global dfBuilder1Time
global getdfBuilderTime

tradingViewTime = 0
getCurrentPriceTime = 0
getFairValueTime = 0
getOptionsTime = 0
dfBuilder1Time = 0
getdfBuilderTime = 0

class StockDetail():

    api_key = 'ADLK0ZK57SB9BJKX'

    def __init__(self, symbol):

        self.symbol = symbol
    # returns a bunch of information like a description of the company, marketcap, 52 week lows and highs, EPS, Dividend, etc.
    def get_company_overview(self):

        url = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol='+self.symbol+'&apikey='+self.api_key
        response = requests.get(url)
        response_json = response.json()

        if len(response_json) == 0:
            return "No data found."

        return response_json

    #retrieves current price of Stock
    def get_quote(self): 

        url = 'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol='+self.symbol+'&apikey='+self.api_key
        response = requests.get(url)
        response_json = response.json()

        if len(response_json) == 0:
            return "No data found."

        response_json = {x[4:].replace(' ','_'): v 
            for x, v in response_json['Global Quote'].items()}

        return response_json
    
    #monthly price data
    def get_monthly_adjusted(self):

        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol='+self.symbol+'&apikey='+self.api_key
        response = requests.get(url)
        response_json = response.json()

        if list(response_json.keys())[0] == 'Error Message':
            return "No data found."

        return response_json

    #live price data from yahooQuery
    def get_price(self):

        ticker = Ticker(self.symbol)
        raw_dict = ticker.price
        df = pd.DataFrame.from_dict(raw_dict)

        return df
    
    #returns earnings trend data
    def get_earnings_trend(self):

        ticker = Ticker(self.symbol)
        raw_dict = ticker.earnings_trend

        output_dict = {}
        output_dict['currentQtr'] = raw_dict[self.symbol]['trend'][0]['epsTrend']
        output_dict['nextQtr'] = raw_dict[self.symbol]['trend'][1]['epsTrend']
        output_dict['currentYr'] = raw_dict[self.symbol]['trend'][2]['epsTrend']
        output_dict['nextYr'] = raw_dict[self.symbol]['trend'][3]['epsTrend']

        return output_dict

    def get_option_chain(self, **kwargs):

        ticker = Ticker(self.symbol)
        df = ticker.option_chain

        if type(df) is str:
            if df == 'No option chain data found':  
                return "No data found."
                
        df.reset_index(inplace=True)
        df = df[df['optionType']=='puts']

        expireDates = df['expiration'].dt.strftime("%Y-%m-%d").unique()    

        df['mid'] = ( ( df['bid'] + df['ask'] ) / 2 ).round(2)
        df['timeToExpire'] = (df['expiration'] - pd.Timestamp.today()).round('1d').dt.days + 1
        df['multiplier'] = 365 / df['timeToExpire']
        df['return'] = ( df['multiplier'] * df['mid'] * 100 ) / ( df['strike'] * 100 )
        df.drop(['optionType', 'contractSymbol', 'currency', 'contractSize', 'lastTradeDate', 'impliedVolatility', 'multiplier'], axis=1, inplace=True)
        df['change'] = df['change'].round(2)
        df['percentChange'] = df['percentChange'].round(2)
        df['return'] = (df['return'] * 100).round(2)

        # Set filters
        if kwargs['expiration'] != 'all' and kwargs['expiration'] != None:
            df = df[ df['expiration'] == kwargs['expiration'] ]

        if kwargs['strikeMin'] != '' and kwargs['strikeMin'] != None:
            df = df[ df['strike'] >= int(kwargs['strikeMin']) ]

        if kwargs['strikeMax'] != '' and kwargs['strikeMax'] != None:
            df = df[ df['strike'] <= int(kwargs['strikeMax']) ]

        if kwargs['returnMin'] != '' and kwargs['returnMin'] != None:
            df = df[ df['return'] >= int(kwargs['returnMin']) ]

        if kwargs['returnMax'] != '' and kwargs['returnMax'] != None:
            df = df[ df['return'] <= int(kwargs['returnMax']) ]

        if kwargs['timeToExpireMin'] != '' and kwargs['timeToExpireMin'] != None:
            df = df[ df['timeToExpire'] >= int(kwargs['timeToExpireMin']) ]

        if kwargs['timeToExpireMax'] != '' and kwargs['timeToExpireMax'] != None:
            df = df[ df['timeToExpire'] >= int(kwargs['timeToExpireMax']) ]

        # Format Columns
        df['expiration'] = df['expiration'].dt.strftime("%Y-%m-%d")

        # Rename Columns
        column_rename = {
            'percentChange': '% Change',
            'openInterest': 'open Int'
        }
        df.rename(columns=column_rename, inplace=True)

        results = {}
        results['columns'] = df.columns.values
        results['expiration'] = expireDates
        results['data'] = df

        return results


#start of tradingView bits
def tradingView():
    global tradingViewTime
    startTime = time.time()
    filter = [
        {
            "left": "volume",
            "operation": "nempty"
        },
        {
            "left": "type",
            "operation": "in_range",
            "right": [
                "stock",
                "dr",
                "fund"
            ]
        },
        {
            "left": "subtype",
            "operation": "in_range",
            "right": [
                "common",
                "foreign-issuer",
                "",
                "etf",
                "etf,odd",
                "etf,otc",
                "etf,cfd"
            ]
        },
        {
            "left": "exchange",
            "operation": "in_range",
            "right": [
                "AMEX",
                "NASDAQ",
                "NYSE"
            ]
        },
        {
            "left": "market_cap_basic",
            "operation": "egreater",
            "right": mktCapMin
        },
        {
            "left": "is_primary",
            "operation": "equal",
            "right": True
        },
        {
            "left": "Stoch.K",
            "operation": "less",
            "right": StochK
        },
        {
            "left": "Stoch.D",
            "operation": "less",
            "right": StochD
        },
        {
            "left": "MACD.macd",
            "operation": "less",
            "right": macd_macd
        },
        {
            "left": "MACD.signal",
            "operation": "less",
            "right": macd_signal
        },
        {
            "left": "dividend_yield_recent",
            "operation": "egreater",
            "right": div_yield_recent
        },
        {
            "left": "Stoch.RSI.K",
            "operation": "less",
            "right": 25
        },
        {
            "left": "Stoch.RSI.D",
            "operation": "less",
            "right": 25
        }
    ]
    options = {
        "lang": "en"
    }
    markets = {
        "america"
    }
    symbols = {
        "query": {
            "types": []
        },
        "tickers": []
    }
    columns = [
        #"logoid",
        "name",
        "description",
        "close",
        "change",
        "change_abs",
        "Recommend.All",
        "market_cap_basic",
        "price_earnings_ttm",
        "earnings_per_share_basic_ttm",
        "sector",
        "earnings_release_date",
        "earnings_release_next_date",
        "dividend_yield_recent"
    ]
    sort = {
        "sortBy": "volume",
        "sortOrder": "desc"
    }
    range = [
        0,
        150
    ]

    post_message = {}
    post_message['filter'] = filter
    post_message['options'] = options
    post_message['symbols'] = symbols
    post_message['columns'] = columns
    post_message['sort'] = sort
    post_message['range'] = range

    payload = json.dumps(post_message)


    url = "https://scanner.tradingview.com/america/scan"
    #     payload = self.payload
    headers = {
        'authority': 'scanner.tradingview.com',
        'accept': 'text/plain, */*; q=0.01',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9',
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'dnt': '1',
        'origin': 'https//www.tradingview.com',
        'referer': 'https//www.tradingview.com/',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="96", "Google Chrome";v="96"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    response_json = response.json()['data']

    df0 = pd.DataFrame.from_dict(response_json)
    df0.drop(columns=['s'], inplace=True)
    df = pd.DataFrame(df0["d"].to_list(), columns=columns)
#     display(df)

    df['change'] = df['change'].round(2)
    df['dividend_yield_recent'] = df['dividend_yield_recent'].round(2)
    df['price_earnings_ttm'] = df['price_earnings_ttm'].round(2)
    df['earnings_per_share_basic_ttm'] = df['earnings_per_share_basic_ttm'].round(2)
#     display(df)
    myListOfTickers = df['name'].tolist()
    executionTime = (time.time() - startTime)
    tradingViewTime = tradingViewTime + executionTime


    return myListOfTickers



#function for getting current price of ticker
def getCurrentPrice(ticker):
    global getCurrentPriceTime
    startTime = time.time()

    yq_ticker = Ticker(ticker)
    #may contain EPS and shares outstanding
    quotes = yq_ticker.quotes[ticker]

    currentPrice = (quotes['bid'] + quotes['ask'])/2
#     print('epsCurrentYear: ', quotes['epsCurrentYear'])
#     print('sharesOutstanding', quotes['sharesOutstanding'])

#     print('important thing is price, date ticker')
#     print(ticker, 'currentPrice: ',  currentPrice)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    groupA = [[now, ticker, currentPrice]]
    groupA_cols = ['dateTime', 'ticker', 'currentPrice']
    livePrice_dfA = pd.DataFrame(groupA, columns = groupA_cols)
    
    executionTime = (time.time() - startTime)
    getCurrentPriceTime = getCurrentPriceTime + executionTime
    
    return livePrice_dfA

global AlphaVantageKey
#basic key AlphaVantageKey = 'XEQ9AFU8KM035KMG'
#premium api key
AlphaVantageKey = 'ZTB6U564ILR50HU3'
def fairValue_hist(ticker):
    global AlphaVantageKey
    global getFairValueTime
    startTime = time.time()
    
    yq_ticker = Ticker(ticker)
    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    url = 'https://www.alphavantage.co/query?function=EARNINGS&symbol='+ticker+'&apikey='+AlphaVantageKey
    r = requests.get(url)
    data = r.json()
    # print(type(data['annualEarnings']))
    # display(data['fannualEarnings'])
    df = pd.DataFrame(data['annualEarnings'])
    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
    filtered_values = np.where((df['fiscalDateEnding'] > '2017-01-01') & (df['fiscalDateEnding'] < '2020-01-01'))
    # display(filtered_values)
    eps1 = df.loc[filtered_values]
#     display(eps1)
    # print(filtered_values)

    testPrice = []
    reportedEPS_cols = []
    pe_cols = []
    testPrice_cols = []

    for index, row in eps1.iterrows():
        #retrieving testPrice from Past
        testPriceDateStart = row['fiscalDateEnding'] + relativedelta(months=+3)
        testPriceDateEnd = row['fiscalDateEnding'] + relativedelta(months=+4)
        df = yq_ticker.history(period='6y', interval='1d')
        priceHistory = pd.DataFrame(df)
        priceHistory = priceHistory.reset_index()
        priceHistory['date']= pd.to_datetime(priceHistory['date'])
        mask = (priceHistory['date'] > testPriceDateStart) & (priceHistory['date'] < testPriceDateEnd)
        avg_price = priceHistory.loc[mask]
        avgTestPrice = avg_price['close'].mean()
        testPrice.append(avgTestPrice)
        currentYear = row['fiscalDateEnding'].strftime("%Y")

        #making all the columns
        reportedEPS_cols.append('reportedEPS' + currentYear)
        pe_cols.append('p/e'+currentYear)
        testPrice_cols.append('avgPrice' + currentYear)
        testdf = pd.DataFrame(testPrice)
#         print('testdf')
#         display(testdf)

    #building vertical dataframe which ends up getting transposed
    eps1.reset_index(drop=True)
    eps1['testPrice'] = testdf.values
    eps1['reportedEPS'] = eps1['reportedEPS'].astype(float)
    eps1['p/e'] = eps1['testPrice']/eps1['reportedEPS']
    eps1['avgP/E'] = eps1['p/e'].mean()
    currentYrEstimate = StockDetail('AAPL').get_earnings_trend()['currentYr']['current']
    eps1['currentYrEstimate'] = currentYrEstimate
    eps1['FairValue'] = eps1['avgP/E'] * eps1['currentYrEstimate']
    
    #bulding long data lists which is what we actually use for fv_df
    data = eps1['reportedEPS'].to_list()
    data.extend(eps1['testPrice'].to_list())
    data.extend(eps1['p/e'].to_list())
    data.append(eps1['currentYrEstimate'].values[0])
    data.append(eps1['FairValue'].values[0])

    #creating final long data row from lists
    fv_df_cols = reportedEPS_cols + testPrice_cols + pe_cols + ['currentYrEstimate', 'FairValue']
#     print(fv_df_cols)
    fv_df = pd.DataFrame(data = [data], columns = fv_df_cols)

    executionTime = (time.time() - startTime)
    getFairValueTime = getFairValueTime + executionTime

    return fv_df


def get_options(ticker, daysOut_start, daysOut_end):
    
    global getOptionsTime
    startTime = time.time()

    #getting options chain
    
    yq_ticker = Ticker(ticker)
    df = pd.DataFrame(yq_ticker.option_chain)
    # display(df)
    df = df.reset_index()
#     print('the columns available from option_chain call')
#     display(df.columns)
    tdf = pd.DataFrame(df[['symbol', 'expiration', 'optionType', 'strike', 'currency', 'lastPrice', 'volume', 'openInterest', 'bid', 'ask', 'lastTradeDate']])    
    tdf = tdf[tdf['optionType'] == 'puts']
    
    #filtering the data down a bit by date
    options_date_start = (pd.to_datetime('today')  + pd.Timedelta(daysOut_start)).strftime('%Y-%m-%d')
    options_date_end = (pd.to_datetime('today')  + pd.Timedelta(daysOut_end)).strftime('%Y-%m-%d')
    tdf = tdf[tdf['expiration'] < options_date_end]
    tdf = tdf[tdf['expiration'] > options_date_start]
    
    
    #calculating return
    tdf['midBid'] = (tdf['bid'] + tdf['ask'])/2
    tdf['today'] = pd.to_datetime("today")
    tdf['time'] = (tdf['expiration'] - tdf['today']).dt.days
    tdf['multiplier'] = 365/tdf['time']
    tdf['return'] = (tdf['multiplier']*tdf['midBid']*100)/(tdf['strike']*100)
#     tdf = tdf[tdf['return'] > .06]

    options_dfC = pd.DataFrame(tdf)
    
    executionTime = (time.time() - startTime)
    getOptionsTime = getOptionsTime + executionTime
    
    return(options_dfC)


#the filtering for only fair value and strike price < (.8 * CP) *.75
def opportunityFiltering(df):
    df = df[df['currentPrice'] < df['FairValue']*.8]
    df = df[df['strike'] < (df['FairValue']*.8)*.75]
    df = df[df['return'] > .06]
    return df
    

#filtering below fair value is currently OFF
def df_builder1(ticker, daysOut_start, daysOut_end):
    global dfBuilder1Time
    startTime = time.time()
    
    #building the dataframe parts A,B, and C
    dfA = getCurrentPrice(ticker)
    dfB = fairValue_hist(ticker)
    dfC = get_options(ticker, daysOut_start, daysOut_end)
    leo_df = pd.DataFrame(dfC)

    #adding the current price data to dataframe
    leo_df['dateTime'] = dfA['dateTime'].values[0]
    leo_df['ticker'] = dfA['ticker'].values[0]
    leo_df['currentPrice'] = dfA['currentPrice'][0]
    
    #the dfB fair value bit
    leo_df[[dfB.columns.values]] = 0
    #loops through and adds the dfB columns to leo 1 column at a time in reverse
    for i in range(1,len(dfB.columns)+1):
        leo_df.iloc[:, -i] = dfB.iloc[:,-i].values[0]
    
    #THIS IS WHAT YOU COMMENT OUT FOR THE SINGLE LOOKUP
    #the filtering for only fair value and strike price < (.8 * CP) *.75
    leo_df = opportunityFiltering(leo_df)

    executionTime = (time.time() - startTime)
    dfBuilder1Time = dfBuilder1Time + executionTime
    return leo_df

def dfClean(bigDF):
    finalCols = ['symbol', 'expiration', 'optionType', 'strike', 'bid', 'ask', 'midBid', 'time', 'return', 'ticker', 'currentPrice']
    colsAdd = list(bigDF.columns[-5:])
    finalCols.extend(colsAdd)
    bigDF = bigDF[finalCols]
    bigDF = pd.DataFrame(bigDF)
    bigDF['expiration'] = bigDF['expiration'].astype(str)
    bigDF = bigDF.dropna()
    return bigDF


def df_builderList(tickerList, daysOut_start, daysOut_end):
    global getdfBuilderTime
    startTime = time.time()
    
    ticker1 = 'AAPL'
    leo_df1 = df_builder1(ticker1, daysOut_start, daysOut_end)
    for ticker in tickerList:
        try:
            print('working on the', ticker, 'df')
            df2append = df_builder1(ticker, daysOut_start, daysOut_end)
            leo_df1 = leo_df1.append(df2append)
        except Exception as e:
            print(e)
            print('we are skipping: ', ticker)
            
    executionTime = (time.time() - startTime)
    getdfBuilderTime = getdfBuilderTime + executionTime
    #cleans the final DataFrame
    leo_df1 = dfClean(leo_df1)
    return leo_df1

def saveAsSheet(bigDF):
    # Auth
    creds, _ = default()
    gc = gspread.authorize(creds)

    spreadsheet_key = '1SDTbhY0sn_ZoVXroFfNrfeH_lq3EnH7M1SSYTc14VAA'
    workbook = gc.open_by_key(spreadsheet_key)
    workbook.values_clear('A1:AD10000')
    workbook.values_update(
      'sheet1',
      params={
          'valueInputOption': 'USER_ENTERED'
      },
      body={
          'values': [bigDF.columns.values.tolist()] + bigDF.values.tolist()
      }
    )

# start calling functions to build df code
# -----------------------------------------------------------------------------------------------------------------------------------
startTime = time.time()

ListOfTickers = tradingView()
print(ListOfTickers)
# ListOfTickers = ['AAPL', 'MSFT', 'GM', 'CCL', 'META', 'BABA', 'IBM', 'AMD', 'NVDA', 'QCOM', 'CMG', 'WMT', 'CAT']
# ListOfTickers = ['AMD']
bigDF = df_builderList(ListOfTickers, daysOut_start, daysOut_end)

# bigDF.to_excel("outputSingle.xlsx")

# saveAsSheet(bigDF)
# start Dash CODE
# -----------------------------------------------------------------------------------------------------------------------------------
# df = pd.read_csv('https://git.io/Juf1t')
df = bigDF

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.Div([
        dcc.Dropdown(id='xaxis')]),
    dbc.Label('Click a cell in the table:'),
    dash_table.DataTable(df.to_dict('records'),[{"name": i, "id": i} for i in df.columns], id='tbl'),
    dbc.Alert(id='tbl_out'),
])

@callback(Output('tbl_out', 'children'), Input('tbl', 'active_cell'))
def update_graphs(active_cell):
    return str(active_cell) if active_cell else "Click the table"

if __name__ == "__main__":
    app.run_server(debug=True)
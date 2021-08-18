import ccxt
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import statsmodels.api as sm
from functools import reduce
from statsmodels.tsa.api import *
import progressbar
import warnings
import time as t

def crypto(trading_pair, figs = False):
    warnings.filterwarnings("ignore")
    df = pd.DataFrame()
    try:
        okex = ccxt.okex()
        poloniex = ccxt.poloniex()

        # collect the candlestick data from Binance
        binance = ccxt.binance()
        # candles = binance.fetch_ohlcv(trading_pair, '1d', okex.parse8601('2017-11-08T00:00:00'), poloniex.parse8601(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        candles = binance.fetch_ohlcv(trading_pair, '1d')
        dates = []
        open_data = []
        high_data = []
        low_data = []
        close_data = []
        for candle in candles:
            dates.append(datetime.fromtimestamp(candle[0] / 1000.0).strftime('%Y-%m-%d %H:%M:%S.%f'))
            open_data.append(candle[1])
            high_data.append(candle[2])
            low_data.append(candle[3])
            close_data.append(candle[4])

        crypto = pd.DataFrame({'Open Price': open_data, 'High Price': high_data,'Low Price': low_data,'Close': close_data}, index = dates)
        df = crypto[['Close']]
        df = df.set_index(pd.to_datetime(df.index))
        df = df.resample('D').mean()

        if figs:
            decomposition = sm.tsa.seasonal_decompose(df, model = 'Addictive') #'multiplicative'
            fig, ax = plt.subplots(figsize = (12, 6))
            ax.grid(True)
            year = mdates.YearLocator(month=1)
            month = mdates.MonthLocator(interval=3)
            year_format = mdates.DateFormatter('%Y')
            month_format = mdates.DateFormatter('%m')
            ax.xaxis.set_minor_locator(month)
            ax.xaxis.grid(True, which = 'minor')
            ax.xaxis.set_major_locator(year)
            ax.xaxis.set_major_formatter(year_format)
            plt.plot(df.index, df['Close'], c='blue')
            plt.plot(decomposition.trend.index, decomposition.trend, c='red')
    except:
       pass
    return df

print('Collecting top 50 cryptos')
crypto_list = pd.read_html('https://coinmarketcap.com/all/views/all/')[2].Symbol.loc[:50]

bar = progressbar.ProgressBar(maxval=len(crypto_list), widgets=[progressbar.Bar('>', '|', '| Obtaining Cryptos'), '...', progressbar.Percentage()])
bar.start()
bar_ = 0
for i in crypto_list:
    vars()[i] = crypto('{}/USDT'.format(i)).rename(columns={'Close':'{}'.format(i)})
    bar_ += 1
    bar.update(bar_)     
bar.finish()

dfs = []
for i in crypto_list:
    if len(vars()[i]) == 500:
        dfs.append(vars()[i])
df_final = reduce(lambda left, right: pd.merge(left, right, left_index = True, right_index = True), dfs)

print('Collecting DONE')

# SHORT VERSION
warnings.filterwarnings("ignore")

coin_req = 'ETC'
days_of_forecast = 1
data = df_final

print('Predicting next day')

model = VAR(data)
model_fit = model.fit()
pred = model_fit.forecast(model_fit.y, steps=days_of_forecast)

pred = pd.DataFrame(pred, columns = df_final.columns, index = (pd.Series(pd.date_range('today', periods=days_of_forecast, freq='D').normalize(), name='Date')).values)
pred.iloc[0] = df_final.iloc[-1]

merged = df_final.append(pred).resample('D').mean()
changes = pd.DataFrame(merged.pct_change(periods = days_of_forecast).iloc[-1].sort_values(ascending=False))
changes.columns = ['Change']

plt.figure(figsize = (18, 6))
plt.plot(df_final[coin_req], label = 'Timeseries of {}'.format(coin_req))
plt.plot(pred[coin_req], 'r', label = 'VAR prediction of {}'.format(coin_req))
plt.grid()
plt.legend()
plt.title('Prediction: {} change by {} %'.format(coin_req, round(float(changes.loc[coin_req].values), 3)))
table = plt.table(cellText=np.round(changes.values, 3), colWidths = [0.5] * len(changes.columns),
      rowLabels = changes.index, cellLoc = 'left', rowLoc = 'left', loc = 8, edges='open') # Adjust table size loc and allignment
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(.5, 1.2);

print('Export to excel')

changes.to_excel(r'Changes{}.xlsx'.format(str(pred.index[0])[:-8]), engine='openpyxl')

print('Done')



from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from datetime import datetime
import yfinance as yf
import pandas as pd

@staticmethod
def get_historical(quote):
    try:
        end = datetime.now()
        start = datetime(end.year-1, end.month, end.day)
        data = yf.download(quote, start=start, end=end)
        df = pd.DataFrame(data=data)
        df.to_csv(''+quote+'.csv')
        if df.empty:
            ts = TimeSeries(key='N6A6QT6IBFJOPJ70', output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol=quote, outputsize='full')
            data = data.reset_index()
            # Keep Required cols only
            df = pd.DataFrame()
            df['Date'] = data['date']
            df['Open'] = data['1. open']
            df['High'] = data['2. high']
            df['Low'] = data['3. low']
            df['Close'] = data['4. close']
            df['Adj Close'] = data['5. adjusted close']
            df['Volume'] = data['6. volume']
            df.to_csv(''+quote+'.csv', index=False)
        return
    except Exception as e:
        print("Could not fetch historical data for {}. Try with another stock symbol.".format(quote))

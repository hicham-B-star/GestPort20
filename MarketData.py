from datetime import datetime

import pandas_datareader as web


class MarketData:
    def __init__(self):
        self.table = self.setDataMarket()

    def setDataMarket(self):
        stocks = ['AAPL', 'AMZN', 'GOOGL', 'FB']
        start = datetime(2019, 1, 1)
        end = datetime(2019, 12, 31)
        return web.DataReader(stocks, 'yahoo', start, end)['Adj Close']

import cv2
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import numpy as np


class TradingGraph:
    '''
    TradingGraph class for visualizing the performance of the agent
    
    '''
    def __init__(self, render_range: int = 100, title: str = "Trading Graph") -> None:
        self.volume = deque(maxlen=render_range)
        self.net_worth = deque(maxlen=render_range)
        self.render_data = deque(maxlen=render_range) # open, high, low, close data for candlestick graph
        self.render_range = render_range

        plt.style.use("ggplot")

        plt.close("all") # close all previous matplotlib plots
        self.fig = plt.figure(figsize=(16, 8))
        self.ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
        self.ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=self.ax1)
        self.ax3 = self.ax1.twinx()

        self.date_format = mpl_dates.DateFormatter("%d-%m-%Y")

    def render(self, date: int, open: float, high: float, low: float, close: float, volume: float, net_worth: float, trades: int) -> None:
        self.volume.append(volume)
        self.net_worth.append(net_worth)

        date = mpl_dates.date2num([pd.to_datetime(date)])[0]
        self.render_data.append([date, open, high, low, close])

        self.ax1.clear()
        candlestick_ohlc(self.ax1, self.render_data, width=0.8, colorup="green", colordown="red", alpha=0.8)

        date_render_range = [d[0] for d in self.render_data]
        self.ax2.clear()
        self.ax2.fill_between(date_render_range, self.volume, 0)

        self.ax3.clear()
        self.ax3.plot(date_render_range, self.net_worth, color="blue")

        self.ax1.xaxis.set_major_formatter(self.date_format)
        self.fig.autofmt_xdate()

        for trade in trades:
            trade_date = mpl_dates.date2num([pd.to_datetime(trade['date'])])[0]
            if trade_date in date_render_range:
                if trade['type'] == 'buy':
                    price = trade['low'] - 10
                    self.ax1.scatter(trade_date, price, color='green', s=100, marker='^')
                else:
                    price = trade['high'] + 10
                    self.ax1.scatter(trade_date, price, color='red', s=100, marker='v')
                    

        self.ax2.set_xlabel("Date")
        self.ax1.set_ylabel("Price")
        self.ax3.yaxis.set_label_position("right")
        self.ax3.yaxis.tick_right()
        self.ax3.set_ylabel("Net Worth")

        self.fig.tight_layout()

        self.fig.canvas.draw()
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Trading Graph", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return

if __name__ == "__main__":
    pass
        
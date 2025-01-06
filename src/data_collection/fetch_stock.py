import yfinance as yf
import pandas as pd

def fetch_tesla_stock():
  tesla = yf.Ticker("TSLA")
  history = tesla.history(period="5y")
  history.reset_index(inplace=True)

  history.to_csv("data/raw/tesla_stock_recent.csv", index=False)
  print(f"Tesla stock data saved. {len(history)} records collected.")

fetch_tesla_stock()
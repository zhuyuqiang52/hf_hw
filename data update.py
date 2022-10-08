import pandas as pd
import yfinance as yf
'''sp_assets = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sym = sp_assets['Symbol'].str.replace('.','-').tolist()
data = yf.download(tickers=sym,start = '2022-09-23',end = '2022-09-28')
adj_close_df = data.iloc[:,:503]
adj_close_df.columns = [i[1] for i in adj_close_df.columns]
adj_close_df.to_csv(r'E:\study\22fall\hf\data\hw3\SP500_components.csv')'''
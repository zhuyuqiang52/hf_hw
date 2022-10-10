import pandas as pd
import yfinance as yf
ticker_list = ['AUDUSD=X','EURJPY=X','EURUSD=X','GBPUSD=X','CAD=X','CHF=X','HKD=X','JPY=X','SGD=X','NZDUSD=X','EURGBP=X']
try:
    FX_df = pd.read_csv(r'E:\study\22fall\hf\data\NR4\FX_G10.csv',index_col=0)
except:
    FX_df = yf.download(tickers=ticker_list,start = '2015-08-09',end = '2022-10-30')
    FX_df.to_csv(r'E:\study\22fall\hf\data\NR4\FX_G10.csv')
#transform
FX_df.drop(axis=0,index=['Date'],inplace=True)
FX_df.index = ['ticker']+FX_df.index[1:].tolist()
col_list = []
for col in FX_df.columns:
    col_list.append(col.split('.')[0])
FX_df.columns = col_list
FX_df =FX_df.T
FX_df.loc[:,'Kind'] = FX_df.index
FX_melt_df = FX_df.melt(id_vars=['ticker','Kind'],var_name=['Date'])
FX_pivot_df = FX_melt_df.pivot_table(index=['Date','ticker'],columns = 'Kind')
FX_pivot_df = FX_pivot_df.reset_index()
FX_pivot_df.set_index('Date',inplace=True)
FX_pivot_df.columns = [i[1] for i in FX_pivot_df.columns]
FX_pivot_df.columns = ['ticker']+FX_pivot_df.columns[1:].tolist()
FX_pivot_df.to_csv(r'E:\study\22fall\hf\data\NR4\FX_G10.csv')
pass

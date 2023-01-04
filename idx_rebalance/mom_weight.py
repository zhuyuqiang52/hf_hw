import pandas as pd
import datetime as dt
import yfinance as yf
from tool_func import *
pfl_df = pd.read_csv(r'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\equal_weights.csv',index_col= 0 )
pfl_abs_df = pfl_df.abs()
pfl_count_df = pfl_abs_df.sum(axis=1)
pfl_df.index = pd.to_datetime(pfl_df.index)
period_int = 60
adj_price_cmb_df = pd.DataFrame()

for j in range(pfl_df.shape[0]):
    pfl_sub_df = pfl_df.iloc[[j],:]
    beg_dt = pfl_sub_df.index[0]
    end_dt= beg_dt+dt.timedelta(days=period_int)
    pre_beg_dt = beg_dt+dt.timedelta(days = -90)
    # download data
    # period rank day+1 trade day -> rank day+1 trade day +60 calendar day
    try:
        adj_close_df = pd.read_csv(rf'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\data\rankday_bkt\close_{pre_beg_dt.strftime("%Y%m%d")}_to_{end_dt.strftime("%Y%m%d")}.csv',index_col=0)
    except:
        #ticker list
        ticker_list = pfl_sub_df.columns.tolist()
        adj_close_df = yf.download(tickers=ticker_list,start = pre_beg_dt.strftime('%Y-%m-%d'),end = end_dt.strftime('%Y-%m-%d'),threads=6)
        adj_close_df = adj_close_df.iloc[:,:len(ticker_list)]
        adj_close_df.columns = [i[1] for i in adj_close_df.columns]
        adj_close_df.to_csv(rf'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\data\rankday_bkt\close_{pre_beg_dt.strftime("%Y%m%d")}_to_{end_dt.strftime("%Y%m%d")}.csv')
    adj_close_df.index = pd.to_datetime(adj_close_df.index)
    adj_ror_df = adj_close_df/adj_close_df.shift(1)-1
    adj_ror_df.dropna(axis=0,how='all',inplace=True)
    adj_ror_df = adj_ror_df.loc[:beg_dt,:]
    mom_df = momentum_factor(adj_ror_df,120)
    pfl_df.iloc[[j],:] = pfl_df.iloc[[j],:]*mom_df.iloc[[-1],:].abs()
pass
pfl_df.to_csv(r'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\mom_weights.csv')


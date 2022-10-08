#%%
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
from tool_func import *
#abbr description:
# pfl: pfl
# d: daily
# ror: rate of return
try:
    adj_close_df = pd.read_csv(r'E:\study\22fall\hf\data\hw1\SP500_components.csv',index_col=0)
except:
    #sp500 components
    sp_assets = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sym = sp_assets['Symbol'].str.replace('.','-').tolist()
    data = yf.download(tickers=sym,start = '2021-08-09',end = '2022-09-11')
    adj_close_df = data.iloc[:,:503]
    adj_close_df.columns = [i[1] for i in adj_close_df.columns]
    adj_close_df.to_csv(r'E:\study\22fall\hf\data\hw1\SP500_components.csv')
#%%
adj_ror_df = adj_close_df/adj_close_df.shift(1)-1
ror_10_treasure_float = 0.03319

#%%

#allocation
def allocate(cash_float:float,weights_array:np.array,price_array:np.array)->tuple: #commission: 0.005/share
    cash_weighted_array = cash_float*weights_array
    share_array = np.floor(cash_weighted_array/(price_array+0.005))
    price_sum_float = np.sum(share_array*price_array+0.005*share_array)
    cash_float -= price_sum_float
    return cash_float,share_array




momentum_df = momentum_factor(adj_ror_df,120)
IC_analysis(momentum_df,adj_ror_df,'spearman')
factor_ret(momentum_df,adj_ror_df)
#%%

#pfl construction
# choose 5 stock with top X factor value with equal weight
nums_int = 5
cash_init_float = 1e8
end_dt_str = '2022-09-08'
bkt_period_int = 252
end_loc_int = np.where(adj_close_df.index==end_dt_str)[0][0]
beg_loc_int = end_loc_int-bkt_period_int
beg_dt_str = dt
tickers_list = momentum_df.iloc[-1,:].sort_values().tail(nums_int+1).index.tolist()[:-1] #last one lack data
weights_array = np.array([1/nums_int]*nums_int)
price_array = adj_close_df.iloc[beg_loc_int,:][tickers_list].values
cash_float,share_array = allocate(cash_init_float,weights_array,price_array)

#%%
# cal pfl mkt val
ror_df = adj_ror_df.iloc[beg_loc_int + 1:end_loc_int+1, :][tickers_list]
ror_array = ror_df.values
accu_ret_array = np.multiply.accumulate(ror_array + 1, axis=0)
stk_mkt_val_array = share_array.reshape(1,-1) * price_array.reshape(1,-1)
mkt_val_array = stk_mkt_val_array * accu_ret_array
mkt_val_array = np.concatenate((stk_mkt_val_array, mkt_val_array), axis=0)
pfl_d_ret_array = cash_init_float*ror_array
pfl_d_ret_cmb_array = pfl_d_ret_array.sum(axis=1).reshape(-1, 1)
partial_sum_pfl_ret_array = np.add.accumulate(pfl_d_ret_array, 0)
pfl_mv_array = np.sum(mkt_val_array, axis=1) + cash_float

# bmk ret
try:
    sp500_adj_close_df = pd.read_csv(r'E:\study\22fall\hf\data\hw1\SP500.csv',index_col=0)
except:
    data = yf.download(tickers='SPY',start = '2021-08-09',end = '2022-09-11')
    sp500_adj_close_df = data.iloc[:,-2]
    sp500_adj_close_df.to_csv(r'E:\study\22fall\hf\data\hw1\SP500.csv')
end_loc_int = np.where(sp500_adj_close_df.index==end_dt_str)[0][0]
beg_loc_int = end_loc_int-bkt_period_int
sp500_adj_ror_df = sp500_adj_close_df/sp500_adj_close_df.shift(1)-1
sp500_adj_ror_df.dropna(axis=0,inplace=True)
sp500_adj_ror_bkt_df = sp500_adj_ror_df.iloc[beg_loc_int + 1:end_loc_int + 1, :]
bmk_d_ror_array = sp500_adj_ror_bkt_df.values.reshape(-1,1)
sp500_accu_ror_array = np.multiply.accumulate(sp500_adj_ror_bkt_df+1,axis=0).values
bmk_ret_array = cash_init_float * sp500_accu_ror_array
bmk_ret_array = np.concatenate((np.array(cash_init_float).reshape(1, 1), bmk_ret_array.reshape(-1, 1)))
bmk_d_ret_array = np.diff(bmk_ret_array, 1, axis=0)
partial_sum_bmk_ret_array = np.add.accumulate(bmk_d_ret_array, 0)

# gen dataset
d_ret_array = np.concatenate((pfl_d_ret_cmb_array, bmk_d_ret_array), axis=1)

# metric ouput
pfl_ror_array = np.matmul(weights_array.reshape(1, -1), ror_array.T)
pfl_sharp_float = sharp(pfl_ror_array)
sp500_adj_ror_bkt_array = sp500_adj_ror_bkt_df.values.reshape(1, -1)
bmk_sharp_float = sharp(sp500_adj_ror_bkt_df.values.reshape(1, -1))
corrs_float = np.corrcoef(d_ret_array.T)
#cal beta
beta_pfl_float = cal_beta(pfl_ror_array,sp500_adj_ror_bkt_array)

print(f'corr: {corrs_float[0][1]}')
print(f'sharp:\npfl: {pfl_sharp_float},\nbenchmark(SP500): {bmk_sharp_float}')
print(f'beta: {beta_pfl_float}')
#%%

#plotting
plt.plot(np.sum(partial_sum_pfl_ret_array,axis=1),label='pfl')
plt.plot(partial_sum_bmk_ret_array,label='benchmark : SP500')
plt.legend()
plt.show()
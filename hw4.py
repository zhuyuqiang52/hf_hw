from tool_func import *
from hw3 import *
import pandas as pd
import yfinance as yf

try:
    spy_close_df = pd.read_csv(r'E:\study\22fall\hf\data\hw4\spy_close.csv',index_col=0)
except:
    data = yf.download(tickers='SPY',start = '2021-08-09',end = '2022-09-30')
    spy_close_df = data.iloc[:,-2]
    spy_close_df.to_csv(r'E:\study\22fall\hf\data\hw4\spy_close.csv')

spy_ror_df = spy_close_df/spy_close_df.shift(1)-1
spy_ror_df.index = pd.to_datetime(spy_ror_df.index)
spy_ror_df.columns = ['SPY']
spy_ror_df.dropna(axis=0,inplace=True)
#%% md
# (A)
# calculation of statistic

asset_val_df = pos.daily_asset_val(dat_df,True)
pfl_ret_df = asset_val_df-asset_val_df.shift(1)
pfl_ret_df.dropna(axis=0,inplace=True)

#daily return plotting
pfl_ret_df.plot()
plt.title('Faily Return')
plt.show()

pfl_ror_df = asset_val_df/asset_val_df.shift(1)-1
pfl_ror_df.dropna(axis=0,inplace=True)
data_df = pd.merge(spy_ror_df,pfl_ror_df,left_index=True,right_index=True).fillna(0)
#IR
IR_series = data_df.iloc[:,1:].apply(axis=0,func = IR,args = (data_df.iloc[:,0],))
corr_series = data_df.iloc[:,1:].corrwith(data_df.iloc[:,0],axis=0)
beta_float = cal_beta(data_df.iloc[:,0].values,data_df.iloc[:,1].values)
sharp_float = sharp(data_df.iloc[:,1].values)
#%% md
#assese
#%% md
# (B) VaR and ETL
VaR_float = VaR(0.75,pfl_ret_df,method_str='series')
ETL_float = ETL(0.75,pfl_ret_df,method_str='series')
max_drawdown_float = max_drawdown(asset_val_df)

pass


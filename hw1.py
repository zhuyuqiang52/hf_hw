import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

#read data
stk_a_df = pd.read_csv(r'E:\study\22fall\hf\data\hw1\ALB.csv')[["Date",'Adj Close']].dropna(axis=0).set_index('Date')
stk_a_df.index = pd.to_datetime(stk_a_df.index)
stk_b_df = pd.read_csv(r'E:\study\22fall\hf\data\hw1\XOM.csv')[['Date','Adj Close']].dropna(axis=0).set_index('Date')
stk_b_df.index = pd.to_datetime(stk_b_df.index)
bmk_df = pd.read_csv(r'E:\study\22fall\hf\data\hw1\sp500.csv')[['Date','Adj Close']].dropna(axis=0).set_index('Date')
bmk_df.index = pd.to_datetime(bmk_df.index)
bmk_df.loc[:,'Adj Close'] = np.float64(bmk_df.loc[:,'Adj Close'])
stk_a_df = stk_a_df.reindex(bmk_df.index)
stk_b_df = stk_b_df.reindex(bmk_df.index)
#return
ror_a_df = stk_a_df/stk_a_df.shift(1)-1
ror_b_df = stk_b_df/stk_b_df.shift(1)-1
ror_bmk_df = bmk_df/bmk_df.shift(1)-1
ror_df = pd.concat([ror_a_df,ror_b_df,ror_bmk_df],axis=1)
ror_10_treasure_float = 0.03319
#allocation
def allocate(cash_float:float,weights_array:np.array,price_array:np.array)->tuple: #commission: 0.005/share
    cash_weighted_array = cash_float*weights_array
    share_array = np.floor(cash_weighted_array/(price_array+0.005))
    price_sum_float = np.sum(share_array*price_array+0.005*share_array)
    cash_float -= price_sum_float
    return cash_float,share_array

def sharp(ror_d_array):
    ror_y_float = np.exp(np.log(1+ror_d_array).sum() / (ror_d_array.shape[1]/252))-1
    std_y_float = np.std(ror_d_array)*np.sqrt(252)
    return (ror_y_float-0.03319)/std_y_float

# parameters
end_str = '2022-09-01'
end_loc_int = np.where(stk_a_df.index==end_str)[0][0]

#backtest last X trade days
period_int = 20
beg_loc_int = end_loc_int-period_int
cash_init_float = 1000000.0
price_array = np.array([stk_a_df.iloc[beg_loc_int,0],stk_b_df.iloc[beg_loc_int,0]])
weights_array = np.array([0.2,0.8])
cash_float,share_array = allocate(cash_init_float,weights_array,price_array)

#cal pfl mkt val
ror_array = ror_df.iloc[beg_loc_int+1:end_loc_int+1,:].values
accu_ret_array = np.multiply.accumulate(ror_array+1,axis=0)
stk_mkt_val_array = share_array*price_array.reshape(1,2)
mkt_val_array = stk_mkt_val_array*accu_ret_array[:,:2]
mkt_val_array = np.concatenate((stk_mkt_val_array,mkt_val_array),axis=0)
pfl_d_ret_array = cash_init_float*ror_array[:,:2]
pfl_d_ret_cmb_array = pfl_d_ret_array.sum(axis=1).reshape(-1,1)
partial_sum_pfl_ret_array = np.add.accumulate(pfl_d_ret_array,0)
pfl_mv_array = np.sum(mkt_val_array,axis=1)+cash_float

#bmk ret
bmk_ret_array = cash_init_float*accu_ret_array[:,2]
bmk_ret_array = np.concatenate((np.array(cash_init_float).reshape(1,1),bmk_ret_array.reshape(-1,1)))
bmk_d_ret_array = np.diff(bmk_ret_array,1,axis=0)
partial_sum_bmk_ret_array = np.add.accumulate(bmk_d_ret_array,0)

#gen dataset
d_ret_array = np.concatenate((pfl_d_ret_cmb_array,bmk_d_ret_array),axis=1)

#metric ouput
porfolio_ror_array = np.matmul(weights_array.reshape(1,2),ror_array[:,:2].T)
pfl_sharp_float = sharp(porfolio_ror_array)
bmk_sharp_float = sharp(ror_array[:,-1].reshape(1,-1))
corrs_float = np.corrcoef(d_ret_array.T)
#print(f'corr: {corrs_float[0][1]}')
#print(f'sharp:\npfl: {pfl_sharp_float},\nbenchmark(SP500): {bmk_sharp_float}')

#plotting
'''plt.plot(np.sum(partial_sum_pfl_ret_array,axis=1),label='pfl')
plt.plot(partial_sum_bmk_ret_array,label='benchmark : SP500')
plt.legend()
plt.show()'''
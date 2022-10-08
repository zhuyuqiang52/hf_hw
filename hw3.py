from tool_func import *
#%% md
#(A) get data
#read data
dat_df = pd.read_csv(r'E:\study\22fall\hf\data\hw3\hw3_dat.csv',index_col=0)
dat_df.index = pd.to_datetime(dat_df.index)
#%% md
# (B) signal generation
ror_df = dat_df/dat_df.shift(1)-1
ror_df.dropna(axis=0,inplace=True,how = 'all')
ror_df.fillna(0,inplace=True)
mom_centered_df = momentum_factor(ror_df,5,centered_func=np.median)
#%% md
# (C) portfolio construction
# first, market cap hedge
long_int = 2
short_int = 2
portfolio_weight_df = signal_sizing(mom_centered_df,'signal',long_nums=long_int,short_nums=short_int)
# initialize position account
# (D)
pos = position(2e6,False)
pos.order_execute(weights_df=portfolio_weight_df.iloc[[-3*5-1],:],price_df=dat_df.iloc[[-3*5-1],:],brokerage_fee_rate_float=0.0001)
pos.order_execute(weights_df=portfolio_weight_df.iloc[[-2*5-1],:],price_df=dat_df.iloc[[-2*5-1],:],brokerage_fee_rate_float=0.0001)
pos.order_execute(weights_df=portfolio_weight_df.iloc[[-1*5-1],:],price_df=dat_df.iloc[[-1*5-1],:],brokerage_fee_rate_float=0.0001)
pos.order_execute(weights_df=portfolio_weight_df.iloc[[0*5-1],:],price_df=dat_df.iloc[[0*5-1],:],brokerage_fee_rate_float=0.0001)
#pos.daily_ret(dat_df,True)
# (E)
#pos.accu_ret_plot()
pass
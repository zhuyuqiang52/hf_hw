import pandas as pd

from tool_func import *
import yfinance as yf
import datetime as dt
import pandas as pd
# ticker load
ticker_df = pd.read_excel(r'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\data\rankday_Chg_ticker.xlsx',sheet_name='RankDay_AddDrop')
#ticker clean
#drop all doesn't contain US
#drop all contain numbers

for i in ticker_df.columns:
    contain_us_series = ticker_df.loc[:,i].str.contains(' US ')
    false_idx_list = contain_us_series.where(contain_us_series.values == False).dropna().index.tolist()
    ticker_df.loc[false_idx_list, i] = np.nan
    begwith_nums_series = ticker_df.loc[:,i].str.startswith(('0','1','2','3','4','5','6','7','8','9'))
    num_idx_list = begwith_nums_series.where(begwith_nums_series.values==True).dropna().index.tolist()
    ticker_df.loc[num_idx_list,i] = np.nan
#signal list
pfl_beg_list = ['20190513','20200511','20210510'] #next trade day after rank day
pfl_list = []
for i in range(3):
    add_list = ticker_df.iloc[:,2*i].dropna().str.split(' ').tolist()
    for idx in range(len(add_list)):
        add_list[idx] = add_list[idx][0]
    drop_list = ticker_df.iloc[:, 2 * i+1].dropna().str.split(' ').tolist()
    for idx in range(len(drop_list)):
        drop_list[idx] = drop_list[idx][0]
    #idx_list =
    pfl_df = pd.DataFrame(index=add_list+drop_list,columns= [dt.datetime.strptime(pfl_beg_list[i],'%Y%m%d')])
    pfl_df.iloc[:len(add_list),:] = 1
    pfl_df.iloc[len(add_list):,:] = -1
    pfl_list.append(pfl_df)

#postion setting
pos = position(2e6,fix_notion_bool=True)

period_int = 90
adj_price_cmb_df = pd.DataFrame()
for pfl_df in pfl_list:
    beg_dt = pfl_df.columns[0]
    end_dt= beg_dt+dt.timedelta(days=period_int)

    # download data
    # period rank day+1 trade day -> rank day+1 trade day +60 calendar day
    try:
        adj_close_df = pd.read_csv(rf'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\data\rankday_bkt\close_{beg_dt.strftime("%Y%m%d")}_to_{end_dt.strftime("%Y%m%d")}.csv',index_col=0)
    except:
        #ticker list
        ticker_list = pfl_df.index.tolist()
        adj_close_df = yf.download(tickers=ticker_list,start = beg_dt.strftime('%Y-%m-%d'),end = end_dt.strftime('%Y-%m-%d'),)
        adj_close_df = adj_close_df.iloc[:,:len(ticker_list)]
        adj_close_df.columns = [i[1] for i in adj_close_df.columns]
        adj_close_df.to_csv(rf'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\data\rankday_bkt\close_{beg_dt.strftime("%Y%m%d")}_to_{end_dt.strftime("%Y%m%d")}.csv')
    adj_close_df.index = pd.to_datetime(adj_close_df.index)
    #drop no data ticker(maybe defunct or not trading)
    adj_close_df.dropna(axis = 1,how = 'all',inplace = True)
    adj_close_df.dropna(axis=0,how='all',inplace=True)
    #drop price under 1 usd
    adj_close_df =adj_close_df.where(adj_close_df>1.0).dropna(axis=1,how = 'any')
    pfl_df = pfl_df.reindex(adj_close_df.columns.tolist())
    #equal weighting
    eq_weights_df = equal_weight(pfl_df)
    eq_weights_df = eq_weights_df.T
    sub_price_df =adj_close_df.iloc[[0],:].dropna(axis=1)
    eq_weights_df = eq_weights_df.reindex(columns = sub_price_df.columns)

    pos.order_execute(eq_weights_df,sub_price_df)
    cover_price_df = adj_close_df.iloc[[-1],:].ffill()
    pos.cover(cover_price_df)
    adj_price_cmb_df = pd.concat([adj_price_cmb_df,adj_close_df],axis= 0)

#plot and stats
#pos.accu_ret_plot()
daily_asset_val_df = pos.daily_asset_val(adj_price_cmb_df,True)

#statsitics
#download SPY
try:
    spy_df = pd.read_csv(rf'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\data\SPY.csv',index_col=0)
except:
    #sp500 components
    ticker_list = ['SPY']
    spy_df = yf.download(tickers=ticker_list,start = daily_asset_val_df.index[0].strftime('%Y-%m-%d'),
                                       end = daily_asset_val_df.index[-1].strftime('%Y-%m-%d'))
    spy_df = adj_close_df.iloc[:,[-2]]
    spy_df.columns = [i[1] for i in spy_df.columns]
    spy_df.to_csv(rf'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\data\SPY.csv')
#daily ret
spy_df.index = pd.to_datetime(spy_df.index)
daily_asset_val_expand_df = daily_asset_val_df.reindex(spy_df.index).ffill()
pnl_df = daily_asset_val_expand_df-pos.init_exposure_float
pnl_df.columns = ['index_rebalance']

#plotting
pnl_df.plot(title = 'PNL',figsize = (40,10))
plt.show()

spy_ror_df = spy_df/spy_df.shift(1)-1
spy_ror_df.dropna(axis=0,inplace = True)
pfl_ror_df = daily_asset_val_expand_df/daily_asset_val_expand_df.shift(1)-1
pfl_ror_df.dropna(axis=0,inplace=True)

IR_float = IR(pfl_ror_df.values,spy_ror_df.values)
Sharp_float = sharp(pfl_ror_df.values)
beta_float = cal_beta(pfl_ror_df.values,spy_ror_df.values)
maxdrawdown_float = max_drawdown(daily_asset_val_df)
VaR_float = VaR(0.99,pnl_df)
ETL_float = ETL(0.99,pnl_df)
annualized_ror_float = annualized_ror(pfl_ror_df.values)
pass








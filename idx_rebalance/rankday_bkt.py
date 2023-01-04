import matplotlib.pyplot as plt
import pandas as pd

from tool_func import *
import yfinance as yf
import datetime as dt
import pandas as pd

RD_path_str = r'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\data\preRankDayTop3k'
idx_path_str = r'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\data\r1000'

rankday_list = ['20000531','20010531','20020531','20030530','20040528','20050527','20060531','20070531','20080530',
                '20090529','20100528','20110531','20120531','20130531','20140530','20150529','20160527','20170512'
                ,'20180511','20190510','20200508','20210507','20220506']
def bkt(b_coef,bd_add_int,end_add_int,weight_type_str,oop = ''):
    if b_coef == 0:
        hedged_str = 'unhedged'
    else:
        hedged_str = 'hedged'
    version_str = oop+hedged_str+'_'+weight_type_str+'_beg_'+str(bd_add_int)+'d_late_end_'+str(end_add_int)+'d_earely'
    try:
        pfl_df = pd.read_csv(r'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\oop_weights\\'+weight_type_str+'.csv',index_col= 0 )
        pfl_df.index = pd.to_datetime(pfl_df.index)
    except:
        #gen add and drop ticker for each year
        year_list = list(range(2000,2023))

        rankday_dt_list = pd.to_datetime(rankday_list).tolist()
        pfl_df = pd.DataFrame()
        for idx in range(19,len(year_list)):
            year = year_list[idx]
            last_year = year-1
            last_idx_df = pd.read_csv(idx_path_str + f'\RIY{last_year}.csv')
            idx_ticker_list = last_idx_df['Ticker'].str.split(' ').tolist()
            RD_df = pd.read_excel(RD_path_str+f'\\{year}PreFriday9.xlsx',header=0)
            cap_list = []
            for i in RD_df['Market Cap'].tolist():
                try:
                    cap_list.append(float(i))
                except:
                    cap_list.append(0)
            RD_df['Market Cap'] = cap_list
            #ticker clean
            RD_U_series = RD_df.loc[:, 'Ticker'].str.contains(' U. ')
            RD_filter_df = RD_df.where(RD_U_series==True).dropna(axis=0,how='all').sort_values('Market Cap',ascending = False)
            RD_T1k_series = RD_filter_df.iloc[:1000,0]
            RD_T1k_list = RD_T1k_series.str.split(' ').tolist()
            for i in range(len(RD_T1k_list)):
                RD_T1k_list[i] = RD_T1k_list[i][0]
            for j in range(len(idx_ticker_list)):
                idx_ticker_list[j] = idx_ticker_list[j][0]
            add_list = list(set(RD_T1k_list).difference(set(idx_ticker_list)))
            drop_list = list(set(idx_ticker_list).difference(set(RD_T1k_list)))
            if len(drop_list)==0:
                print('error')
            #gen signal
            if pfl_df.empty:
                pfl_df = pfl_df.reindex(columns = add_list+drop_list)
                pfl_df.loc[rankday_dt_list[idx]] = 0
                pfl_df.loc[rankday_dt_list[idx],add_list] = 1
                pfl_df.loc[rankday_dt_list[idx], drop_list] = -1
            else:
                pfl_sub_df = pd.DataFrame(index = [rankday_dt_list[idx]],columns = add_list+drop_list)
                pfl_sub_df.loc[rankday_dt_list[idx],add_list] = 1
                pfl_sub_df.loc[rankday_dt_list[idx], drop_list] = -1
                pfl_df = pd.concat([pfl_df,pfl_sub_df],axis=0)
        pfl_df.to_csv(r'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\equal_weights_pre9.csv')
        pfl_df.index = pd.to_datetime(pfl_df.index)
        pfl_df.fillna(0, inplace=True)
    #pfl_df = pd.read_csv(r'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\marketcap_weights.csv',index_col=0)
    pfl_df.index = pd.to_datetime(rankday_list[-4:])
    pfl_df.fillna(0,inplace=True)
    pass

    #spy
    spy_df = pd.read_csv(rf'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\data\SPY_CLOSE.csv',index_col=0)
    spy_df.columns = ['SPY_IDX']
    spy_df.index = pd.to_datetime(spy_df.index)
    spy_df = spy_df.sort_index(ascending=True)
    spy_ror_df = spy_df/spy_df.shift(1)-1
    spy_ror_df.dropna(axis=0,inplace = True)
    #postion setting
    pos = position(2e6,fix_notion_bool=True)


    period_int = 60
    adj_price_cmb_df = pd.DataFrame()
    for j in range(pfl_df.shape[0]):
        pfl_sub_df = pfl_df.iloc[[j],:]
        beg_dt = pfl_sub_df.index[0]
        end_dt= beg_dt+dt.timedelta(days=period_int+40)
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
        beg_loc_int = np.where(adj_close_df.index == beg_dt)[0]

        # alter start date
        beg_dt = adj_close_df.index[beg_loc_int+bd_add_int][0]


        #drop no data ticker(maybe defunct or not trading)
        adj_close_df.dropna(axis = 1,how = 'all',inplace = True)
        adj_close_df.dropna(axis=0,how='all',inplace=True)
        #drop price under 1 usd
        adj_close_df =adj_close_df.where(adj_close_df>1.0).dropna(axis=1,how = 'any')
        #combine spy
        adj_close_df = pd.merge(adj_close_df,spy_df,left_index=True,right_index=True,how='left')
        pfl_sub_df = pfl_sub_df.reindex(columns = adj_close_df.columns.tolist())
        #equal weighting
        eq_weights_df = equal_weight(pfl_sub_df)
        sub_price_df =adj_close_df.loc[[beg_dt],:].dropna(axis=0)
        eq_weights_df = eq_weights_df.reindex(columns = sub_price_df.columns).fillna(0)

        eq_weights_df.index = [beg_dt]

        #pre rank day pfl for beta calculation
        pre_price_df = adj_close_df.loc[:beg_dt, :]
        #pre_price_df = adj_close_df.loc[:end_dt,:]
        pre_ror_df = pre_price_df/pre_price_df.shift(1)-1
        pre_ror_df.dropna(axis=0,inplace=True)
        pre_weights_df = eq_weights_df.reindex(pre_ror_df.index).bfill().ffill()
        pre_pfl_ror_df = pre_weights_df*pre_ror_df
        pre_pfl_ror_df.fillna(0,inplace=True)
        pre_pfl_ror_sum_df = pre_pfl_ror_df.sum(axis=1).to_frame()
        pre_pfl_ror_sum_df = pre_pfl_ror_sum_df.dropna(axis=0)
        sub_spy_ror_df = spy_ror_df.reindex(pre_pfl_ror_sum_df.index)
        pfl_beta_float = cal_beta(pre_pfl_ror_sum_df.loc[:beg_dt,:].values,sub_spy_ror_df.loc[:beg_dt,:].values)
        #aft_pfl_beta_float = cal_beta(pre_pfl_ror_sum_df.loc[beg_dt:,:].values,sub_spy_ror_df.loc[beg_dt:,:].values)
        beta_weighted_float = b_coef*pfl_beta_float #suppose to be zero
        #adjust weights
        eq_weights_adj_df = eq_weights_df.copy()
        eq_weights_adj_df.loc[:,'SPY_IDX'] = [-beta_weighted_float]

        pos.order_execute(eq_weights_adj_df,sub_price_df)
        cover_price_df = adj_close_df.iloc[[-1-end_add_int],:].ffill() # can alter cover date
        pos.cover(cover_price_df)
        adj_price_cmb_df = pd.concat([adj_price_cmb_df,adj_close_df.loc[beg_dt:cover_price_df.index[-1],:]],axis= 0)
    #plot and stats
    #pos.accu_ret_plot()
    daily_asset_val_df = pos.daily_asset_val(adj_price_cmb_df,True)



    #daily ret

    daily_asset_val_expand_df = daily_asset_val_df.reindex(spy_df.loc[:end_dt,:].index).ffill().dropna(axis=0)
    pnl_df = daily_asset_val_expand_df-pos.init_exposure_float
    pnl_df.columns = ['index_rebalance']

    output_path_str = r'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\output\basic'
    #plotting
    ax = pnl_df.plot(title = 'PNL',figsize = (20,10)).get_figure()
    ax.show()
    ax.savefig(output_path_str+'\\pnl_plot_'+version_str+'png')
    #statsitics

    pfl_ror_df = daily_asset_val_df/daily_asset_val_df.shift(1)-1
    pfl_ror_df.dropna(axis=0,inplace=True)
    spy_ror_df = spy_ror_df.reindex(pfl_ror_df.index)
    pfl_ror_exp_df = daily_asset_val_expand_df/daily_asset_val_expand_df.shift(1)-1
    pfl_ror_exp_df.dropna(axis=0,inplace=True)
    spy_ror_exp_df = spy_ror_df.reindex(pfl_ror_df.index).fillna(0)
    IR_float = IR(pfl_ror_df.values,spy_ror_df.values)
    Sharp_float = sharp(pfl_ror_df.values)
    beta_float = cal_beta(pfl_ror_df.values,spy_ror_df.values)
    maxdrawdown_float = max_drawdown(daily_asset_val_df)
    VaR_float = VaR(0.99,pnl_df.diff().dropna(axis=0))
    ETL_float = ETL(0.99,pnl_df.diff().dropna(axis=0))
    annualized_ror_float = annualized_ror(pfl_ror_exp_df.values)[0]
    annualized_vol_float = vol(pfl_ror_df)
    statistics_df = pd.DataFrame(index = ['IR','Sharpe','beta(to SPY)','max_drawdown','VaR','ETL','annualized_ror','annualized_vol'],data=[IR_float,Sharp_float,beta_float,maxdrawdown_float,VaR_float,ETL_float,annualized_ror_float,annualized_vol_float])

    pos.cost_df.to_csv(output_path_str+'\\cost_'+version_str+'.csv')
    daily_asset_val_expand_df.to_csv(output_path_str+'\\daily_asset_val_'+version_str+'.csv')
    pnl_df.to_csv(output_path_str+'\\pnl_'+version_str+'.csv')
    pfl_ror_exp_df.to_csv(output_path_str+'\\ror_'+version_str+'.csv')
    statistics_df.to_csv(output_path_str+'\\stat_'+version_str+'.csv')

if __name__ == '__main__':
    b_coef = 1
    bd_add_int = 10
    end_add_int = 5
    hedged_str = ''
    weight_type_str = ''
    #for weight_type_str in ['equal_weights','marketcap_weights','reciprocal_weights']:
    for weight_type_str in ['basic_inverse_weights']:
        for bd_add_int in [0]:
            for end_add_int in [15]:
                for b_coef in [1]:
                    try:
                        bkt(b_coef,bd_add_int,end_add_int,weight_type_str,'oop_pre_')
                    except:
                        continue

pass








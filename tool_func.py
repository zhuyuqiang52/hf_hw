import numpy as np
import statsmodels.api as sm
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
#glbal parameter
ror_10_treasure_float = 0.03319

class position:
    def __init__(self,gross_exposure_float,fix_notion_bool:bool=True):
        self.fix_notion_bool = fix_notion_bool
        self.init_exposure_float = gross_exposure_float
        self.gross_exposure_float = gross_exposure_float
        self.cash_float = 0
        self.asset_val_float = gross_exposure_float
        self.position = pd.DataFrame()
        self.proceed_from_short_sale = 0
        self.last_val_array = 0
        self.asset_list = []
        self.ret_left = 0
        self.ret_left_list = [0]
        self.cash_list = []
        self.proceed_from_short_sale_list = []
        self.long_val_float = 0
        self.long_val_list = []
        self.cover_dates_list = []
        self.cost_df = pd.DataFrame(columns=['brokerage_fee','stock_loan_fee','capital_gain_tax'])
    def asset_val(self,price_df,fix_notion_bool:bool):
        price_array = price_df.reindex(columns = self.position.columns).fillna(0).values
        cur_val_array = self.position.iloc[-1,:].values*price_array
        val_chg_float = np.sum(cur_val_array)-np.sum(self.last_val_array)
        #tax
        if val_chg_float > 41675 and val_chg_float<459750:
            self.cost_df.loc[price_df.index[0], :] = [0, 0, 0.15 * val_chg_float]
            val_chg_float = (1-0.15)*val_chg_float
        elif val_chg_float>459750:
            self.cost_df.loc[price_df.index[0], :] = [0, 0, 0.2 * val_chg_float]
            val_chg_float = (1-0.2)*val_chg_float
        self.gross_exposure_float = np.sum(np.abs(cur_val_array))
        self.asset_val_float = self.cash_float+self.long_val_float+self.ret_left+self.proceed_from_short_sale+val_chg_float
        #update long val
        self.long_val_float = np.sum(cur_val_array[cur_val_array>0])

        if fix_notion_bool:
            left_float = self.asset_val_float-self.init_exposure_float-self.ret_left
            self.ret_left += left_float
            self.long_val_float-=left_float
            self.ret_left_list.append(self.ret_left)
    def order_execute(self,weights_df = None,price_df = None,
                      slippage_rate_float=0.0008,
                      brokerage_fee_rate_float = 0.0001,
                      stock_loan_fee_rate_float = 0.0001,
                      proceed_interest_rate_float=0.01,
                      cover = False):
        if self.position.shape[0]>0:
            last_position_df = self.position.iloc[[-1],:]
            last_position_df.index = price_df.index
        execute_price_array = price_df.values*(1+np.random.normal(slippage_rate_float,0.0001,size=price_df.shape[1]))
        execute_price_array = np.nan_to_num(execute_price_array,nan=0.0)# suppose slippage follow a normal distribution
        execute_price_df = pd.DataFrame(execute_price_array,index = price_df.index,columns = price_df.columns)

        # update asset val
        if self.position.shape[0] >= 1:
            self.asset_val(execute_price_df, self.fix_notion_bool)

        # switch between fix notion and actual asset
        if weights_df is not None:
            abs_weight_df = np.abs(weights_df)
            abs_weight_df = abs_weight_df/np.sum(abs_weight_df.values) # weights constructed considering gross_exposure

            if self.fix_notion_bool:
                exposure_weighted_df = self.init_exposure_float*(1-2*brokerage_fee_rate_float)*abs_weight_df*np.sign(weights_df.values)
            else:
                exposure_weighted_df = self.asset_val_float*(1-2*brokerage_fee_rate_float)*abs_weight_df*np.sign(weights_df.values)
            exposure_weighted_df = exposure_weighted_df.fillna(0)   # multiply original sign to distinguish long/short exposure
            #per_cent_float = exposure_weighted_df.abs().iloc[0,-1]/exposure_weighted_df.iloc[:,:-1].abs().sum(axis=1)
            shares_df = np.floor_divide(exposure_weighted_df,execute_price_array*(1+(brokerage_fee_rate_float+stock_loan_fee_rate_float)/2))
        elif cover:
            #for cover
            shares_df = -last_position_df.copy()
            self.cover_dates_list.append(shares_df.index[0])
        if self.position.shape[0]>0:
            shares_increment_df =shares_df +last_position_df
        else:
            shares_increment_df = shares_df
        shares_increment_df.fillna(0,inplace=True)
        #update last val array
        val_array = shares_increment_df.values * execute_price_df.reindex(columns = shares_increment_df.columns).fillna(0).values
        self.last_val_array = val_array
        #update cash and gross exposure
        self.gross_exposure_float = np.sum(np.abs(val_array))
        # update long pos val
        self.long_val_float = val_array[val_array > 0].sum()
        self.long_val_list.append(self.long_val_float)
        # update short sale proceed
        self.proceed_from_short_sale = -val_array[val_array < 0].sum()
        self.proceed_from_short_sale_list.append(self.proceed_from_short_sale)
        # cash doesn't take account of transaction fee
        self.cash_float = self.asset_val_float-self.long_val_float-self.proceed_from_short_sale-self.ret_left
        #update postion
        self.position = pd.concat([self.position,shares_increment_df])
        self.position.fillna(0,inplace=True)
        #update cash
        if self.position.shape[0]>1: # if have altered position, then there exists a t least 2 position log
            ex_price_array = execute_price_df.reindex(columns = self.position.columns).fillna(0).values
            val_chg_array = (self.position.iloc[-1,:]-self.position.iloc[-2,:]).values*ex_price_array
            #exposure_chg_float = val_chg_array.sum()
            brokerage_fee_float = np.abs(val_chg_array).sum()*brokerage_fee_rate_float # brokerage fee for all transactions
            #stock_loan_fee cal
            day_gap_int = (self.position.index[-1]-self.position.index[-2]).days
            stock_loan_fee_float = self.proceed_from_short_sale_list[-2]*(np.power(1+stock_loan_fee_rate_float-proceed_interest_rate_float,day_gap_int/252)-1) # stock loan fee - proceed interest
            #subtract cost
            self.cash_float -= stock_loan_fee_float+brokerage_fee_float
            self.cost_df.loc[price_df.index[0]] = [brokerage_fee_float,stock_loan_fee_float,0]
        else:
            brokerage_fee_float = np.abs(val_array).sum() * brokerage_fee_rate_float
            self.cost_df.loc[price_df.index[0]] = [brokerage_fee_float, 0, 0]
            # subtract cost
            self.cash_float -= brokerage_fee_float
        self.asset_val_float = self.long_val_float+self.proceed_from_short_sale+self.cash_float+self.ret_left
        self.asset_list.append(self.asset_val_float)
        self.cash_list.append(self.cash_float)

    def cover(self,price_df):
        self.order_execute(cover=True,price_df = price_df)
    def accu_ret_plot(self):
        asset_val_array = np.array(self.asset_list)-self.init_exposure_float
        plt.plot(asset_val_array,label='dollar return')
        plt.legend()
        plt.show()

    def daily_asset_val(self,price_df,all_price_bool=False):
        price_df.fillna(0,inplace=True)
        time_idx = self.position.index.tolist()
        price_idx = np.array(price_df.index.tolist())
        beg_loc_int = np.where(price_idx == time_idx[0])[0][0]
        if all_price_bool:
            end_loc_int = price_df.shape[0]-1
        else:
            end_loc_int = np.where(price_idx == time_idx[-1])[0][0]
        price_sub_df = price_df.iloc[beg_loc_int:end_loc_int+1,:]
        share_df = self.position.reindex(price_sub_df.index).ffill() #previous val fill
        price_sub_df = price_sub_df.reindex(columns = share_df.columns)
        val_df = share_df*price_sub_df
        val_df.to_csv('Position_Val.csv')
        if len(self.cover_dates_list)>0:
            #sepearely cal accumulate val chg
            sub_val_chg_list = []
            for date in self.cover_dates_list:
                date = date+dt.timedelta(days=1)
                sub_df = val_df.loc[:date]
                sub_chg_df = sub_df.diff().fillna(0)
                sub_chg_sum_df =sub_chg_df.sum(axis=1)
                sub_chg_accu_df = np.add.accumulate(sub_chg_sum_df)
                sub_val_chg_list.append(sub_chg_accu_df)
                val_df = val_df.loc[date:]
            val_chg_accu_df = pd.concat(sub_val_chg_list).to_frame()
        else:
            val_chg_df = val_df.diff().fillna(0)
            val_chg_sum_df = val_chg_df.sum(axis=1)
            val_chg_accu_df = np.add.accumulate(val_chg_sum_df).to_frame()
        #add left ret, cash
        val_chg_accu_df.loc[self.cover_dates_list,:] = 0
        if self.fix_notion_bool:
            other_array = np.array(self.cash_list)+np.array(self.proceed_from_short_sale_list)+np.array(self.long_val_list)+np.array(self.ret_left_list)
        else:
            other_array = np.array(self.cash_list)+np.array(self.proceed_from_short_sale_list)+np.array(self.long_val_list)
        other_array = other_array.reshape(-1,1)
        other_df = pd.DataFrame(data = other_array,index = time_idx)
        other_df = other_df.reindex(price_sub_df.index).ffill()
        asset_val_df = pd.merge(val_chg_accu_df,other_df,left_index=True,right_index=True)
        asset_val_df.loc[price_df.index[beg_loc_int],'0_y'] = self.init_exposure_float
        asset_val_df.sort_index(ascending=True,inplace=True)
        asset_val_df.fillna(0,inplace=True)
        asset_val_sum_df = pd.DataFrame(asset_val_df.sum(axis=1))
        return asset_val_sum_df

def max_drawdown(pfl_ret_df):
    drawdown_df = (np.maximum.accumulate(pfl_ret_df)-pfl_ret_df)/np.maximum.accumulate(pfl_ret_df)
    drawdown_df.fillna(0,inplace=True)
    loc_end_int = np.argmax(drawdown_df)
    loc_beg_int = np.argmax(pfl_ret_df.iloc[:loc_end_int,0])
    max_drawdown_float = pfl_ret_df.iloc[loc_beg_int,0]-pfl_ret_df.iloc[loc_end_int,0]
    return max_drawdown_float

def VaR_std_m(confidence_level_float,asset_val_df,period_int = 1):
    daily_ror_df = asset_val_df/asset_val_df.shift(1)-1
    daily_ror_df.fillna(0,inplace=True)
    std_float = daily_ror_df.std(axis=0)
    std_y_float = std_float*np.sqrt(252)
    z_quantile_float = norm.ppf(confidence_level_float,loc = 0, scale = 1)
    var_float = (std_y_float*z_quantile_float*np.sqrt(period_int/252))*asset_val_df.iloc[0,0]
    return var_float

def VaR_series(confidence_level_float,daily_ret_df,period_int=1):
    confidence_level_float *=100
    daily_ret_array = daily_ret_df.values.copy()
    Var_float = np.percentile(daily_ret_array,100-confidence_level_float,interpolation='midpoint')*period_int
    return Var_float

def VaR(confidence_level_float,daily_ret_df,period_int = 1,method_str='series'):
    if method_str=='series':
        return VaR_series(confidence_level_float,daily_ret_df,period_int)
    elif method_str=='std':
        return VaR_std_m(confidence_level_float, daily_ret_df,period_int)

def ETL(confidence_level_float,daily_ret_df,period_int = 1,method_str='series'):
    VaR_float = VaR(confidence_level_float,daily_ret_df,period_int,method_str)
    etl_ret_df = daily_ret_df[daily_ret_df<=VaR_float]
    etl_float = etl_ret_df.mean().values[0]
    return etl_float

def allocate(cash_float:float,weights_array:np.array,price_array:np.array)->tuple: #commission: 0.005/share
    cash_weighted_array = cash_float*weights_array
    share_array = np.floor(cash_weighted_array/(price_array+0.005))
    price_sum_float = np.sum(share_array*price_array+0.005*share_array)
    cash_float -= price_sum_float
    return cash_float,share_array

def IR(pfl_r_array:np.array,bmk_r_array:np.array,annualized_bool=True)->float:
    std_float = np.std(pfl_r_array - bmk_r_array)
    IR_float = np.mean(pfl_r_array-bmk_r_array)/std_float
    if annualized_bool:
        IR_float *=np.sqrt(252)
    return IR_float

def sharp(ror_d_array,annualized_bool=True):
    ror_d_array = ror_d_array.reshape(-1,1)
    std_y_float = np.std(ror_d_array)
    return np.mean(ror_d_array-ror_10_treasure_float/252)/std_y_float*np.sqrt(252)

def IC_analysis(factor_df,ror_df,method_str = 'pearson'):
    factor_df.dropna(axis=0,inplace=True,how='all')
    #ror multiindex & shift -1
    ror_shift_df = ror_df.shift(-1)
    ror_shift_df = ror_shift_df.reindex(factor_df.index,axis = 0)
    corrs_df = ror_shift_df.corrwith(factor_df,axis = 1,method = method_str)
    corrs_df.dropna(axis=0,inplace=True,how='all')
    corrs_array = corrs_df.values
    IC_mean_float = np.mean(corrs_array)
    IC_std_float = np.std(corrs_array)
    IC_IR_float = IC_mean_float/IC_std_float
    IC_pos_rate_float = len(corrs_array[corrs_array>0])/len(corrs_array)
    print(f'\nIC Analysis\nIC_mean: {IC_mean_float},\nIC_std: {IC_std_float},\nIC_IR: {IC_IR_float},\n(IC>0)%: {IC_pos_rate_float}')
def vol(ror_df,annualized = True):
    vol_float = ror_df.std().values
    if annualized:
        vol_float *= np.sqrt(252)
    return  vol_float[0]
def factor_ret(factor_df,ror_df,chg_period_int = 20):
    factor_df.dropna(axis=0, inplace=True, how='all')
    #20 days accumulate ror
    ror_accu_df = np.log(ror_df+1)
    ror_accu_df = np.exp(ror_accu_df.rolling(chg_period_int,axis=0).sum())-1
    ror_shift_df = ror_accu_df.shift(-chg_period_int).dropna(axis=0)
    idx_intersect_list = np.intersect1d(factor_df.index.tolist(),ror_shift_df.index.tolist()).tolist()
    ror_shift_df = ror_shift_df.reindex(idx_intersect_list, axis=0)
    factor_df = factor_df.reindex(idx_intersect_list,axis=0)
    #reg for factor ret
    factor_ret_list = []
    t_list = []
    for i in range(factor_df.shape[0]):
        X_array = factor_df.iloc[i,:].values
        y_array = ror_shift_df.iloc[i,:].values
        # nan detect
        nan_locX_array = np.where(np.isnan(X_array)==True)
        nan_locy_array = np.where(np.isnan(y_array)==True)
        nan_loc_array = np.union1d(nan_locy_array,nan_locX_array)
        X_array = np.delete(X_array,nan_loc_array)
        y_array = np.delete(y_array,nan_loc_array)
        #add constant
        X_array = sm.add_constant(X_array)
        model = sm.OLS(y_array,X_array)
        result = model.fit()
        t_val_float = result.tvalues[1]
        coef_float = result.params[1]
        factor_ret_list.append(coef_float)
        t_list.append(t_val_float)
    t_abs_array = np.abs(t_list)
    t_abs_mean_float = np.mean(t_abs_array)
    t_abs_significant_float = len(t_abs_array[t_abs_array>2])/len(t_abs_array)
    t_mean_float = np.mean(t_list)
    ret_mean_float = np.mean(factor_ret_list)
    print(f'\nRegression Analysis\nabs tvalue mean: {t_abs_mean_float}\nsignificance of t value( abs>2): {t_abs_significant_float}'
          f'\nt value mean: {t_mean_float}\nret mean: {ret_mean_float}')

def factor_ror(factor_df,ror_df,chg_period_int = 20)->pd.DataFrame():
    factor_df.dropna(axis=0, inplace=True, how='all')
    #20 days accumulate ror
    ror_accu_df = np.log(ror_df+1)
    ror_accu_df = np.exp(ror_accu_df.rolling(chg_period_int,axis=0).sum())-1
    ror_shift_df = ror_accu_df.shift(-chg_period_int).dropna(axis=0)
    idx_intersect_list = np.intersect1d(factor_df.index.tolist(),ror_shift_df.index.tolist()).tolist()
    ror_shift_df = ror_shift_df.reindex(idx_intersect_list, axis=0)
    factor_df = factor_df.reindex(idx_intersect_list,axis=0)
    real_dt_idx = [i+dt.timedelta(chg_period_int) for i in idx_intersect_list]
    #reg for factor ret
    factor_ret_list = []
    for i in range(factor_df.shape[0]):
        X_array = factor_df.iloc[i,:].values
        y_array = ror_shift_df.iloc[i,:].values
        # nan detect
        nan_locX_array = np.where(np.isnan(X_array)==True)
        nan_locy_array = np.where(np.isnan(y_array)==True)
        nan_loc_array = np.union1d(nan_locy_array,nan_locX_array)
        X_array = np.delete(X_array,nan_loc_array)
        y_array = np.delete(y_array,nan_loc_array)
        #add constant
        X_array = sm.add_constant(X_array)
        model = sm.OLS(y_array,X_array)
        result = model.fit()
        coef_float = result.params[1]
        factor_ret_list.append(coef_float)
    factor_ror_df = pd.DataFrame(factor_ret_list,index=real_dt_idx)
    return factor_ror_df

def cal_beta(stk_ror_array,bmk_ror_array)->float:
    X_array = sm.add_constant(bmk_ror_array.reshape(-1, 1))
    y_array = stk_ror_array .reshape(-1, 1)
    model = sm.OLS(y_array, X_array)
    result = model.fit()
    beta_float = result.params[1]
    return beta_float

def cal_multibeta(stk_ror_array,factor_ror_list)->float:
    ror_array = np.concatenate(factor_ror_list,axis=1)
    X_array = sm.add_constant(ror_array - ror_10_treasure_float)
    y_array = (stk_ror_array - ror_10_treasure_float).reshape(-1, 1)
    model = sm.OLS(y_array, X_array)
    result = model.fit()
    beta_array = result.params[1:]
    return beta_array

def annualized_ror(d_ror_array)->float:
    d_ror_array = d_ror_array+1
    period_int = len(d_ror_array)
    accu_ror_float = np.multiply.accumulate(d_ror_array)[-1]
    annualized_ror_float = np.power(accu_ror_float,252/period_int)
    annual_ror_float = annualized_ror_float-1

    return annual_ror_float

# factor cal func
def momentum_factor(ror_df,window_int,centered_func=None):
    #centered func can be any numpy func
    ror_df = np.log(ror_df+1)
    momentum_factor_df = np.exp(ror_df.rolling(window_int,axis=0,min_periods = 1).sum())-1
    momentum_factor_df.dropna(axis=0,how='all',inplace=True)
    if centered_func is not None:
        subtract_val_df = momentum_factor_df.apply(axis=1,func=centered_func,result_type='broadcast')
        momentum_factor_centered_df = momentum_factor_df - subtract_val_df
        return momentum_factor_centered_df
    return momentum_factor_df

def signal_weight(factor_df,long_nums = None,short_nums = None):
    if long_nums is None or short_nums is None:
        factor_postive_sum_df = factor_df.where(factor_df >= 0).sum(axis=1)
        factor_negative_sum_df = factor_df.where(factor_df < 0).sum(axis=1)
        factor_weight_neg_df = factor_df.where(factor_df >= 0).div(factor_postive_sum_df, axis=0).fillna(0)
        factor_weight_pos_df = factor_df.where(factor_df < 0).div(-factor_negative_sum_df, axis=0).fillna(0)
        factor_weight_df = factor_weight_pos_df + factor_weight_neg_df
        return factor_weight_df
    else:
        if long_nums+short_nums>factor_df.shape[1]:
            try:
                raise Exception('stocks nums sum larger than the universe of stocks.')
            except Exception as e:
                print(e)
    # quantile value
    arg_sort_df = np.argsort(factor_df,axis=1)
    K_small_loc_df = arg_sort_df.iloc[:,short_nums]
    K_large_loc_df = arg_sort_df.iloc[:,-long_nums]
    row_list = list(range(factor_df.shape[0]))
    boundary_df = pd.DataFrame([factor_df.iloc[i,[K_large_loc_df[i],K_small_loc_df[i]]].values for i in row_list])

    #only weight long nums and short nums assets
    factor1_df = factor_df.sub(boundary_df.iloc[:,0].values,axis=0)
    factor1_df = factor1_df.where(factor1_df>0).add(boundary_df.iloc[:,0].values,axis=0).fillna(0)
    factor2_df = factor_df.sub(boundary_df.iloc[:,1].values,axis=0)
    factor2_df = factor2_df.where(factor2_df<0).add(boundary_df.iloc[:,1].values,axis=0).fillna(0)
    factor_df = factor1_df+factor2_df

    factor_postive_sum_df = factor_df.where(factor_df>0).sum(axis=1)
    factor_negative_sum_df = factor_df.where(factor_df < 0).sum(axis=1)
    factor_weight_neg_df = factor_df.where(factor_df >= 0).div(factor_postive_sum_df,axis=0).fillna(0)
    factor_weight_pos_df = factor_df.where(factor_df < 0).div(-factor_negative_sum_df, axis=0).fillna(0)
    factor_weight_df = factor_weight_pos_df+factor_weight_neg_df
    return factor_weight_df

def equal_weight(signal_df):
    value_count_df = signal_df.T.value_counts()
    try:
        pos_count_int = value_count_df[1]
    except:
        pos_count_int = 0
    try:
        neg_count_int = value_count_df[-1]
    except:
        neg_count_int = 0
    if pos_count_int>0:
        signal_df[signal_df>0] = signal_df[signal_df>0]/pos_count_int
    if neg_count_int>0:
        signal_df[signal_df<0] = signal_df[signal_df<0]/neg_count_int
    return signal_df

def signal_sizing(factor_df,method_str = 'signal',long_nums = None,short_nums = None):
    if method_str == 'signal':
        factor_weight_df = signal_weight(factor_df,long_nums,short_nums)
        return factor_weight_df


def size_factor(mcap_df):
    return mcap_df.log()
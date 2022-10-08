import pandas as pd
import matplotlib.pyplot as plt
fx_df = pd.read_csv(r'E:\study\22fall\hf\data\NR4\FX.csv')
fx_df.loc[:,'range'] = fx_df.loc[:,'High']-fx_df.loc[:,'Low']
range_df = fx_df.loc[:,['Date','ticker','range']].pivot_table(index = 'Date',columns='ticker',values='range')
ibar_df = range_df-range_df.shift(1)
ibar_df = ibar_df.where(ibar_df<0)*0+1
ibar_df.fillna(0,inplace=True)
ibar_df = ibar_df.stack().reset_index()
# NR period parameter
period_int = 4
accu_min_df = range_df.rolling(window=period_int-1).min().shift(1)
accu_min_stack_df = accu_min_df.stack()
accu_min_stack_df = accu_min_stack_df.reset_index()
fx_nr_df = pd.merge(fx_df,accu_min_stack_df,left_on=['Date','ticker'],right_on =['Date','ticker'])
fx_nr_df = pd.merge(fx_nr_df,ibar_df,left_on=['Date','ticker'],right_on =['Date','ticker'])
fx_nr_df.columns = fx_nr_df.columns[:-2].tolist()+['min_range','ibar']
fx_nr_df.loc[:,'NR4_Ibar'] = fx_nr_df.loc[:,'range']-fx_nr_df.loc[:,'min_range']
fx_nr_df.loc[:,'NR4_Ibar'] = fx_nr_df.loc[:,['NR4_Ibar']].where(fx_nr_df.loc[:,'NR4_Ibar']>0)*0
fx_nr_df.loc[:,'NR4_Ibar'] = fx_nr_df.loc[:,['NR4_Ibar']].fillna(1)
fx_nr_df.loc[:,'NR4_Ibar'] *= fx_nr_df.loc[:,'ibar']

#breakout
fx_nr_df.loc[:,'last_NR4Ibar'] = fx_nr_df.loc[:,'NR4_Ibar'].shift(1)
fx_nr_df.loc[:,'high_break'] = fx_nr_df.loc[:,'High']-fx_nr_df.loc[:,'High'].shift(11)
fx_nr_df.loc[:,['high_break']] = fx_nr_df.loc[:,['high_break']].where(fx_nr_df.loc[:,['high_break']]<=0)*0
fx_nr_df.loc[:,'high_break'].fillna(1,inplace=True)
fx_nr_df.loc[:,'low_break'] = fx_nr_df.loc[:,'Low']-fx_nr_df.loc[:,'Low'].shift(11)
fx_nr_df.loc[:,['low_break']] = fx_nr_df.loc[:,['low_break']].where(fx_nr_df.loc[:,['low_break']]>=0)*0
fx_nr_df.loc[:,'low_break'].fillna(-1,inplace=True)
fx_nr_df.loc[:,'short_sig'] = fx_nr_df.loc[:,'low_break']*fx_nr_df.loc[:,'last_NR4Ibar']
fx_nr_df.loc[:,'long_sig'] = fx_nr_df.loc[:,'high_break']*fx_nr_df.loc[:,'last_NR4Ibar']
fx_nr_df.loc[:,'sig'] = fx_nr_df.loc[:,'short_sig']+fx_nr_df.loc[:,'long_sig']

# On Day 5, If price move above NR4 bar's high, Long at the high price; If price falls below NR4's low, short at NR4's low; Exit the trade at that days's close(or next k days's close)
fx_nr_df.loc[:,'last_high'] = fx_nr_df.loc[:,'High'].shift(11)
fx_nr_df.loc[:,'last_low'] = fx_nr_df.loc[:,'Low'].shift(11)
fx_nr_df.loc[:,'last_high'] = fx_nr_df.loc[:,'last_high']*fx_nr_df.loc[:,'sig']
fx_nr_df.loc[:,'last_high'] = fx_nr_df.loc[:,'last_high'].where(fx_nr_df.loc[:,'last_high']>0).fillna(0)
fx_nr_df.loc[:,'last_low'] = fx_nr_df.loc[:,'last_low']*fx_nr_df.loc[:,'sig']
fx_nr_df.loc[:,'last_low'] = -fx_nr_df.loc[:,'last_low'].where(fx_nr_df.loc[:,'last_low']<0).fillna(0)
fx_nr_df.loc[:,'execute_price'] = fx_nr_df.loc[:,'last_high']+fx_nr_df.loc[:,'last_low']

#simulation part
# Dont consider fees, exit at day5
simu_df = fx_nr_df.copy()
simu_df.loc[:,'ror'] = simu_df.loc[:,'sig']*(simu_df.loc[:,'Close']-simu_df.loc[:,'execute_price'])/simu_df.loc[:,'execute_price']

#ror_distribution
plt.hist(simu_df.loc[:,'ror'].dropna(axis=0),rwidth=0.001)
plt.show()
usd_eur_df = fx_nr_df.where(fx_nr_df['ticker']=='EURUSD=X').dropna(axis=0)

pass
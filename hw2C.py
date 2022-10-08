from hw1c import *
api_str = '6325d3de0991b4.12164383'
import urllib,json

#%% md
#DIRECT HEDGING
#%%
#d_ret stands for daily return = init_cash * daily ror
dr_hedged_pfl_ret_array = pfl_d_ret_cmb_array-bmk_d_ret_array*beta_pfl_float
partial_sum_dr_hedged_pfl_ret_array = np.add.accumulate(dr_hedged_pfl_ret_array)
# plotting
plt.plot(np.sum(partial_sum_pfl_ret_array,axis=1),label='pfl')
plt.plot(partial_sum_bmk_ret_array,label='benchmark : SP500')
plt.plot(partial_sum_dr_hedged_pfl_ret_array,label = 'direct hedging')
plt.legend()
plt.show()
#%% md
# INDIRECT HEDGING SElECTION
#%%
end_dt_str = '20220917'
beg_dt_str = '20210917'
#mcap
try:
    mcap_df = pd.read_csv(r'E:\study\22fall\hf\data\mcap.csv')
except:
    mcap_list = []
    for ticker_str in tickers_list:
        url_str = 'https://eodhistoricaldata.com/api/historical-market-cap/'+ticker_str+'.US?api_token=' + api_str
        response = urllib.request.urlopen(url_str)
        data = json.loads(response.read())
        data = pd.DataFrame(data).T.set_index('date')
        mcap_list.append(data)
    mcap_df = pd.concat(mcap_list)
    mcap_df.to_csv(r'E:\study\22fall\hf\data\mcap.csv')
pass
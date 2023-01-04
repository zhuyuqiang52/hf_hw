import pandas as pd

path_str = r'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\data\rd_idx'
preRD_path_str = r'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\data\preRankDayTop3k'
outsiders_3k = []
outsiders_1k = []
outsiders_1k_to_3k = []
year = 2021
for i in [1,9]:
    cmpL_df = pd.read_excel(path_str+'\\'+str(int(year))+'rankday.xlsx',header=2)\
        .sort_values('Market Cap',ascending=False).iloc[:1000,:]
    cmpR_df = pd.read_excel(preRD_path_str+r'\\'+str(int(year))+'PreFriday'+str(i)+'.xlsx',header =2)\
                  .sort_values('Market Cap',ascending=False).iloc[:1000,:]

    cmpL_ticker_list = cmpL_df['Ticker'].tolist()
    cmpL_tk_list = []
    cmpR_ticker_list = cmpR_df['Ticker'].tolist()
    cmpR_tk_list = []
    for L_ticker in cmpL_ticker_list:
        try:
            l_tk_sp = L_ticker.split(' ')
        except:
            continue
        cmpL_tk_list.append(l_tk_sp[0])
    for R_ticker in cmpR_ticker_list:
        try:
            r_tk_sp = R_ticker.split(' ')
        except:
            continue
        cmpR_tk_list.append(r_tk_sp[0])
    count = 0
    L_in_R_list = cmpL_tk_list.copy()
    for i in cmpR_tk_list:
        if i in cmpL_tk_list:
            count+=1
            cmpL_tk_list.remove(i)
    print(f"Pre {i}th friday {year} percent in rank day: {count}/{len(cmpR_tk_list)}: {count/len(cmpR_tk_list)}")

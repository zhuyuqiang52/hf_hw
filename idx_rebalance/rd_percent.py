import pandas as pd

path_str = r'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\data\rd_idx'
r1000_path_str = r'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\data\preRankDayTop3k'
outsiders_3k = []
outsiders_1k = []
outsiders_1k_to_3k = []
years = list(range(2000,2018))
for year in years:
    idx_df = pd.read_excel(r1000_path_str+'\\'+str(int(year))+'rankday.xlsx',header=2).sort_values('Market Cap',ascending=False)
    rd_df = pd.read_excel(r1000_path_str+'\\'+str(int(year))+'PreFriday9.xlsx',header=2).sort_values('Market Cap',ascending=False)

    rd_ticker_list = rd_df['Ticker'].tolist()
    rd_tk_list = []
    idx_ticker_list = idx_df['Ticker'].tolist()
    idx_list = []
    for r_ticker in rd_ticker_list:
        try:
            r_tk_sp = r_ticker.split(' ')
        except:
            continue
        rd_tk_list.append(r_tk_sp[0])
    for i_ticker in idx_ticker_list:
        try:
            i_tk_sp = i_ticker.split(' ')
        except:
            continue
        idx_list.append(i_tk_sp[0])
    count = 0
    idx_3k_list = idx_list.copy()
    for i in rd_tk_list[:-2]:
        if i in idx_3k_list:
            count+=1
            idx_3k_list.remove(i)
    idx_3k_list.sort()
    outsiders_3k.append(idx_3k_list)
    print(f"out of top 3000 {year} percent: {count}/{len(idx_ticker_list)}: {count/len(idx_ticker_list)}")

    idx_1k_list = idx_list.copy()
    count = 0
    for i in rd_tk_list[:-1000]:
        if i in idx_1k_list:
            count+=1
            idx_1k_list.remove(i)
    idx_1k_list.sort()
    outsiders_1k.append(idx_1k_list)
    dif_1k_3k = set(idx_1k_list).difference(set(idx_3k_list))
    outsiders_1k_to_3k.append(dif_1k_3k)
    print(f"out top 1000 {year} percent: {count}/{len(idx_ticker_list)}: {count/len(idx_ticker_list)}\n")

outsiders_1k_df = pd.DataFrame(outsiders_1k,index=years)
outsiders_3k_df = pd.DataFrame(outsiders_3k,index=years)
outsiders_1k_to_3k_df = pd.DataFrame(outsiders_1k_to_3k,index = years)
'''outsiders_3k_df.to_excel('outsiders_3k.xlsx')
outsiders_1k_df.to_excel('outsiders_1k.xlsx')
outsiders_1k_to_3k_df.to_excel('outsiders_1k_to_3k.xlsx')'''
pass
import pandas as pd
import os
import re
import numpy as np


def add_drop(new_idx_df, old_idx_df, year):
    old_ticker_list = old_idx_df.loc[:, 'Ticker']
    new_ticker_list = new_idx_df.loc[:, 'Ticker']
    add_list = list(set(new_ticker_list).difference(set(old_ticker_list)))
    drop_list = list(set(old_ticker_list).difference(set(new_ticker_list)))
    Chg_df = pd.DataFrame(columns=['add', 'drop'])
    max_len_int = max(len(add_list), len(drop_list))
    Chg_df.loc[:, 'add'] = add_list + ['empty'] * (max_len_int-len(add_list))
    Chg_df.loc[:, 'drop'] = drop_list + ['empty'] * (max_len_int-len(drop_list))
    Chg_df.index = [year-1] * max_len_int
    return Chg_df

def op1():
    path_str = r'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\data'
    idx_list = os.listdir(path_str)
    Chg_df = pd.DataFrame()
    for year in range(2019, 2023):
        for s in idx_list:
            if len(re.findall(r'.+' + str(year) + '.+', s)) != 0:
                new_file = re.findall(r'.+' + str(year) + '.+', s)[0]
            if len(re.findall(r'.+' + str(year - 1) + '.+', s)) != 0:
                old_file = re.findall(r'.+' + str(year - 1) + '.+', s)[0]
        new_df = pd.read_excel(path_str+'\\'+new_file)
        old_df = pd.read_excel(path_str+'\\'+old_file)
        change_df = add_drop(new_df, old_df,year)
        Chg_df = pd.concat([Chg_df,change_df],axis=0)
    Chg_df.to_excel(path_str+'\\'+'Constituents_Chg.xlsx')

def top3kChg(year,file_path_str):
    idx_list = os.listdir(file_path_str)
    files_list = []
    for s in idx_list:
        if len(re.findall(str(year) + '.+', s)) != 0:
            files_list += re.findall(str(year) + '.+', s)
    Chg_list = []
    for file_idx in range(1,len(files_list)-1):
        new_df = pd.read_excel(file_path_str + '\\' + files_list[file_idx],header=2)
        old_df = pd.read_excel(file_path_str + '\\' + files_list[file_idx-1],header =2)
        change_df = add_drop(new_df, old_df, year)
        Chg_list.append(change_df)
    return Chg_list
def op2():
    file_path_str = r'C:\Users\zhuyu\PycharmProjects\hf_hw\idx_rebalance\data\eq_cap'
    Chg_list = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
    for year in range(2019,2023):
        change_list = top3kChg(year,file_path_str)
        for i in range(len(change_list)):
            Chg_list[i] = pd.concat([Chg_list[i],change_list[i]],axis=0)
    Chg_df = pd.DataFrame()
    for chg in Chg_list:
        Chg_df = pd.concat([Chg_df,chg.reset_index()],axis=1)
    Chg_df.to_excel(file_path_str+'\\'+'Chg_bef_announcement.xlsx')
    pass





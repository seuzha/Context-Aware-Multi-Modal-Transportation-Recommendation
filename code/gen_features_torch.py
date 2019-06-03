# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import StratifiedKFold

def read_profile_data():
    '''may want to use the average embedding to represent
    the embedding of missing pids?'''
    profile_data = pd.read_csv('../data/profiles.csv')
    profile_na = np.zeros(67)
    profile_na[0] = -1
    profile_na = pd.DataFrame(profile_na.reshape(1, -1))
    profile_na.columns = profile_data.columns
    profile_data = profile_data.append(profile_na)
    return profile_data

def merge_raw_data():
    tr_queries = pd.read_csv('../data/train_queries.csv')
    te_queries = pd.read_csv('../data/test_queries.csv')
    tr_plans = pd.read_csv('../data/train_plans.csv')
    te_plans = pd.read_csv('../data/test_plans.csv')

    tr_click = pd.read_csv('../data/train_clicks.csv')

    tr_data = tr_queries.merge(tr_click, on='sid', how='left')
    tr_data = tr_data.merge(tr_plans, on='sid', how='left')
    tr_data = tr_data.drop(['click_time'], axis=1)
    tr_data['click_mode'] = tr_data['click_mode'].fillna(0)

    te_data = te_queries.merge(te_plans, on='sid', how='left')
    te_data['click_mode'] = -1

    data = pd.concat([tr_data, te_data], axis=0, sort = False)
    data = data.drop(['plan_time'], axis=1)
    data = data.reset_index(drop=True)
    print('total data size: {}'.format(data.shape))
    return data


def gen_od_feas(data):
    data['o1'] = data['o'].apply(lambda x: float(x.split(',')[0]))
    data['o2'] = data['o'].apply(lambda x: float(x.split(',')[1]))
    data['d1'] = data['d'].apply(lambda x: float(x.split(',')[0]))
    data['d2'] = data['d'].apply(lambda x: float(x.split(',')[1]))
    data = data.drop(['o', 'd'], axis=1)
    return data

def drop_duplicate_mode(mode_list, distance_list, price_list, eta_list):
    '''to be called in gen_plan_feas() '''
    tmp_mode_dict={}
    duplicate_ind = -1
    for tmp_ind, tmp_val in enumerate(mode_list):
        if tmp_val in tmp_mode_dict:
            duplicate_ind = tmp_ind
            break
        else:
            tmp_mode_dict[tmp_val]=tmp_ind
    if duplicate_ind != -1:
        mode_list.pop(duplicate_ind)
        distance_list.pop(duplicate_ind)
        price_list.pop(duplicate_ind)
        eta_list.pop(duplicate_ind)
    return mode_list, distance_list, price_list, eta_list

def gen_plan_core(data):
    '''core part of obtaining the long from plan data
    '''
    plan_long_list = []
    for i, [sid, pid, plan, click_mode, weekday, hour] in tqdm(enumerate(data[['sid', 'pid', 'plans', 'click_mode', 'weekday', 'hour']].values)):
        try:
            cur_plan_list = json.loads(plan)
        except:
            cur_plan_list = []

        if len(cur_plan_list) == 0:
            pass
        else:
            for tmp_dict in cur_plan_list:
                if tmp_dict['price']=='':
                    tmp_dict['price'] = np.nan
                plan_long_list.append([sid, pid, tmp_dict['transport_mode'], click_mode, tmp_dict['price'],\
                                        tmp_dict['eta'], tmp_dict['distance'], weekday, hour])

    df_plan_long = pd.DataFrame(plan_long_list, columns = ['sid', 'pid', 'transport_mode', 'click_mode',\
                                    'price', 'eta', 'distance', 'weekday', 'hour'])
    #drop duplicates
    df_plan_long = df_plan_long.drop_duplicates(subset=['sid', 'pid', 'transport_mode'], keep = 'first')

    return df_plan_long

def gen_average_price(data_long):
    df_plan_long = data_long.copy()
    df_plan_long['price_per_distance'] = df_plan_long['price']/df_plan_long['distance']
    df_price_hour = (df_plan_long.loc[:, ['price_per_distance', 'hour']]
                            .groupby('hour', as_index = False)['price_per_distance'].median()
                            .rename(columns = {'price_per_distance':'price_per_distance_hour'})
                            )
    df_price = (df_plan_long.loc[:, ['transport_mode', 'price_per_distance', 'hour']]
                            .groupby(['transport_mode', 'hour'], as_index = False)['price_per_distance'].median()
                            .merge(df_price_hour, on = 'hour', how = 'left')
                            .set_index(['transport_mode', 'hour'])
                            )
    df_price['price_per_distance'] = df_price['price_per_distance'].fillna(df_price['price_per_distance_hour'])
    price_dict= df_price.drop('price_per_distance_hour', axis =1).to_dict()
    return price_dict

def gen_plan_feas(data, price_dict):
    n = data.shape[0]
    mode_avail_list = np.zeros((n, 12))
    mode_rank_list = np.zeros((n, 12))
    max_dist, min_dist, mean_dist, std_dist = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    max_price, min_price, mean_price, std_price = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    max_eta, min_eta, mean_eta, std_eta = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    mode_texts = []
    max_num_mode = 0
    for i, [plan, hour] in tqdm(enumerate(data[['plans', 'hour']].values)):
        try:
            cur_plan_list = json.loads(plan)
        except:
            cur_plan_list = []

        if len(cur_plan_list) == 0:
            mode_avail_list[i, 0] = 1
            mode_rank_list[i, 0] = 9

            max_dist[i] = np.nan
            min_dist[i] = np.nan
            mean_dist[i] = np.nan
            std_dist[i] = np.nan

            max_price[i] = np.nan
            min_price[i] = np.nan
            mean_price[i] = np.nan
            std_price[i] = np.nan

            max_eta[i] = np.nan
            min_eta[i] = np.nan
            mean_eta[i] = np.nan
            std_eta[i] = np.nan

            mode_texts.append('word_null')
        else:
            distance_list = []
            price_list = []
            eta_list = []
            mode_list = []
            for tmp_dict in cur_plan_list:
                distance_list.append(tmp_dict['distance'])
                if tmp_dict['price'] == '':
                    price_list.append(price_dict['price_per_distance'][(tmp_dict['transport_mode'], hour)] * tmp_dict['distance'])
                else:
                    price_list.append(tmp_dict['price'])
                eta_list.append(tmp_dict['eta'])
                mode_list.append(tmp_dict['transport_mode'])

            mode_texts.append(
                ' '.join(['word_{}'.format(mode) for mode in mode_list]))
            #drop duplicate index
            mode_list, distance_list, price_list, eta_list = drop_duplicate_mode(mode_list, distance_list, price_list, eta_list)
            max_num_mode = max(len(mode_list), max_num_mode)
            mode_list = np.array(mode_list, dtype='int')
            distance_list = np.array(distance_list)
            price_list = np.array(price_list)
            eta_list = np.array(eta_list)
            # The following assignment does not matter even if there are duplicates in mode_list
            mode_avail_list[i, mode_list] = 1
            for index, trans_mode in enumerate(mode_list):
                mode_rank_list[i, trans_mode] = 9-index
            distance_sort_idx = np.argsort(distance_list)
            price_sort_idx = np.argsort(price_list)
            eta_sort_idx = np.argsort(eta_list)

            max_dist[i] = distance_list[distance_sort_idx[-1]]
            min_dist[i] = distance_list[distance_sort_idx[0]]
            mean_dist[i] = np.mean(distance_list)
            std_dist[i] = np.std(distance_list)

            max_price[i] = price_list[price_sort_idx[-1]]
            min_price[i] = price_list[price_sort_idx[0]]
            mean_price[i] = np.mean(price_list)
            std_price[i] = np.std(price_list)

            max_eta[i] = eta_list[eta_sort_idx[-1]]
            min_eta[i] = eta_list[eta_sort_idx[0]]
            mean_eta[i] = np.mean(eta_list)
            std_eta[i] = np.std(eta_list)

    feature_data = pd.DataFrame(mode_rank_list)
    feature_data.columns = ['mode_rank_{}'.format(i) for i in range(12)]
    feature_data['max_dist'] = max_dist
    feature_data['min_dist'] = min_dist
    feature_data['mean_dist'] = mean_dist
    feature_data['std_dist'] = std_dist

    feature_data['max_price'] = max_price
    feature_data['min_price'] = min_price
    feature_data['mean_price'] = mean_price
    feature_data['std_price'] = std_price

    feature_data['max_eta'] = max_eta
    feature_data['min_eta'] = min_eta
    feature_data['mean_eta'] = mean_eta
    feature_data['std_eta'] = std_eta

    mode_svd = gen_mode_context_feas(mode_texts)
    # data = pd.concat([data, feature_data, mode_svd], axis=1)
    data = pd.concat([data, feature_data], axis=1)

    data = data.drop(['plans'], axis=1)
    print('max num of mode per session is:', max_num_mode)
    return data, mode_avail_list

def gen_mode_context_feas(mode_texts):
    n_encoding = 6
    tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_vec = tfidf_enc.fit_transform(mode_texts)
    svd_enc = TruncatedSVD(n_components= n_encoding, random_state=2019)
    mode_svd = svd_enc.fit_transform(tfidf_vec)
    # may want to check the signular values to determine
    #1) whether to include that in the feature space
    #2) optimize n_components
    mode_svd = pd.DataFrame(mode_svd)
    mode_svd.columns = ['mode_avail_repre_{}'.format(i) for i in range(n_encoding)]
    return mode_svd

def feature_transform(data):
    ''' may want to check if fillna by mean helps for the following cols'''

    transform_fea_list = ['max_dist',	'min_dist',	'mean_dist', 'std_dist', 'max_price', \
                                'min_price', 'mean_price', 'std_price',	'max_eta', 'min_eta', 'mean_eta', 'std_eta']
    #normalize the features by dividing the standard deviation
    for col in transform_fea_list:
        # data[col] = data[col].fillna(data[col].mean())
        data[col] = data[col]/np.nanstd(data[col])
        data[col] = data[col]/np.nanstd(data[col])

    for col in transform_fea_list:
        data['sqr_'+col] = np.square(data[col])
        data['sqrt_'+col] = np.sqrt(data[col])
    return data

def gen_basic_profile_repre(data):
    profile_data = read_profile_data()
    n_profile_enc = 30
    x = profile_data.drop(['pid'], axis=1).values
    svd = TruncatedSVD(n_components=n_profile_enc, random_state=2019)
    svd_x = svd.fit_transform(x)
    df_profile_repre = pd.DataFrame(svd_x)
    df_profile_repre.columns = ['profile_repre_{}'.format(i) for i in range(n_profile_enc)]
    ave_repre = list(df_profile_repre.mean().values)
    df_profile_repre['pid'] = profile_data['pid'].values
    df_profile_repre = df_profile_repre.append(pd.DataFrame([ave_repre + [-2]], columns = df_profile_repre.columns))
    #pid =-1 corresponds to empty plan list; the click mode should be 0 or -1
    #pid =-2 corresponds to simply missing pid but with plan list; the click mode can be -1, 0, 1, ..., 11.
    data['pid']= np.where(data['pid'].isnull(), np.where(data['mode_rank_0']>0, -1, -2), data['pid'])
    drop_cols = ['mode_rank_'+ str(i) for i in range(12)]
    data = data.drop(drop_cols, axis =1)
    data = data.merge(df_profile_repre, on='pid', how='left')
    return data

def gen_time_feas(data):
    data['req_time'] = pd.to_datetime(data['req_time'])
    data['weekday'] = data['req_time'].dt.dayofweek
    data['hour'] = data['req_time'].dt.hour
    data = data.drop(['req_time'], axis=1)
    return data

def gen_profile_avail_mode_repre(data, mode_avail_list):
    n_encoding = 10
    len_doc = data.shape[0]
    df_mode_avail = pd.DataFrame(mode_avail_list, index = data['pid'].values).reset_index()
    cols = ['mode_avail_' + str(i) for i in range(12)]
    df_mode_avail.columns = ['pid'] + cols
    df_mode_avail = df_mode_avail.groupby('pid', as_index=False)[cols].sum()

    doc_freq = df_mode_avail[cols].sum().values
    inver_doc_freq = [np.log((len_doc+1)/(doc_freq[i] +1))+1 for i in range(12)]
    for index, col in enumerate(cols):
        df_mode_avail[col]=df_mode_avail[col] * inver_doc_freq[index]

    x = df_mode_avail.loc[:, cols].values
    svd = TruncatedSVD(n_components=n_encoding, random_state=2019)
    svd_x = svd.fit_transform(x)
    df_profile_avail_mode_repre = pd.DataFrame(svd_x)
    df_profile_avail_mode_repre.columns = ['profile_avail_mode_repre_{}'.format(i) for i in range(n_encoding)]
    df_mode_avail = pd.concat([df_mode_avail, df_profile_avail_mode_repre], axis=1)
    df_mode_avail = df_mode_avail.drop(cols, axis =1)
    data = data.merge(df_mode_avail, on = 'pid', how = 'left')
    return data

def gen_wide_data(data, price_dict):
    data, mode_avail_list = gen_plan_feas(data, price_dict)
    data = feature_transform(data)
    data = gen_basic_profile_repre(data)
    data = gen_profile_avail_mode_repre(data, mode_avail_list)
    return data

def gen_one_hot_encoding(df, col):
    tmp_df = pd.get_dummies(df[col], prefix = col)
    df = pd.concat([df, tmp_df], axis =1)
    df = df.drop(col, axis =1)
    return df

def gen_rank(df, cols):
    grouped = df.groupby('sid')
    for col in cols:
        df[col+'_rank']=grouped[col].rank()
    df['position_rank'] = df.reset_index().groupby('sid')['index'].rank()
    return df

def gen_long_data(data_long, data_wide, price_dict):
    price_df = pd.DataFrame.from_dict(price_dict).reset_index()
    price_df.columns = ['transport_mode', 'hour', 'price_per_distance']
    data_long = data_long.merge(price_df, on = ['transport_mode', 'hour'], how = 'left')
    data_long['price'] = data_long['price'].fillna(data_long['price_per_distance'] * data_long['distance'])
    data_long = (data_long.drop(['pid','weekday', 'hour', 'click_mode','price_per_distance'], axis =1)
                          .merge(data_wide, on = 'sid')
                          .pipe(gen_one_hot_encoding, 'weekday')
                          .pipe(gen_one_hot_encoding, 'hour')
                          .pipe(gen_one_hot_encoding, 'transport_mode')
                          .pipe(gen_rank, ['price', 'eta', 'distance']))
    print(data_long.isnull().sum().sum())

    return data_long

def merge_data():
    '''check a clever way of filling missing prices;
        and fillnas
        why are there so many sids with min price ==0?
    '''
    data = merge_raw_data().loc[:10000,:]
    # data = merge_raw_data()
    data = gen_od_feas(data)
    data = gen_time_feas(data)
    data_long = gen_plan_core(data)
    price_dict = gen_average_price(data_long)
    data_wide = gen_wide_data(data, price_dict)
    data_long = gen_long_data(data_long, data_wide, price_dict)
    print(data_long.shape)
    return data_long

if __name__ == '__main__':
    data_long = merge_data()
    # data_long has 4.33 G; may consider reducing its size
    data_long.to_csv('../input_torch/data_long.csv')

# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans


def read_profile_data():
    '''may want to use the average embedding to represent
    the embedding of missing pids?'''
    profile_data = pd.read_csv('../data/profiles.csv')
    for pid in [-1, -2]:
        if pid ==-1:
            #####
            # what if using profile_data.mean()
            #####
            # profile_na = np.zeros(67)
            profile_na = profile_data.mean().values
        else:
            profile_na = profile_data.mean().values
        profile_na[0] = pid
        profile_na = pd.DataFrame(profile_na.reshape(1, -1))
        profile_na.columns = profile_data.columns
        profile_data = profile_data.append(profile_na)
    profile_data = gen_profile_cluster(profile_data)
    return profile_data

def gen_profile_cluster(profile_data):
    '''further cluster the pid data;
    note that in K-means; the obtained cluster starts from zero
    '''
    X = profile_data.iloc[:,1:].values
    print('start of k-means')
    kmeans = KMeans(n_clusters=100, random_state=2019, n_init=10).fit(X)
    profile_data['cluster_id'] = kmeans.labels_
    print('end of k-means')
    return profile_data

def gen_weather_data():
    weather = pd.read_json('../data/weather.json')
    df_weather = weather.T.reset_index()
    df_weather.columns = ['date', 'max_temp', 'min_temp', 'weather', 'wind']
    weather_dict = {}
    for i, item in enumerate(df_weather['weather'].unique()):
        weather_dict[item]=i
    df_weather['weather'] = df_weather['weather'].replace(weather_dict)
    df_weather['date'] = '2018-' + df_weather['date']
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    df_weather[['max_temp', 'min_temp', 'weather','wind']] = df_weather[['max_temp', 'min_temp', 'weather','wind']].astype(int)
    return df_weather

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

    data = pd.concat([tr_data, te_data], axis=0)
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
    # try k-means
    for col in ['o1', 'o2', 'd1', 'd2']:
        X= data[col].values.reshape(-1,1)
        print('start of k-means')
        kmeans = KMeans(n_clusters=100, random_state=2019).fit(X)
        data['cate_'+col] = kmeans.labels_
        print('end of k-means')
    return data

def drop_duplicate_mode(mode_list, distance_list, price_list, eta_list, eta_per_dist_list):
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
        eta_per_dist_list.pop(duplicate_ind)

    return mode_list, distance_list, price_list, eta_list, eta_per_dist_list

def gen_average_price(data):
    '''compute average price per unit distance
        may be further adapted to obtain the plan data in the long format
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

    max_eta_per_dist, min_eta_per_dist, mean_eta_per_dist, std_eta_per_dist = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    max_price_per_dist, min_price_per_dist, mean_price_per_dist, std_price_per_dist = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    min_dist_mode, max_dist_mode, min_price_mode, max_price_mode, min_eta_mode, max_eta_mode, first_mode, last_mode = np.zeros(
                    (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    # need to convert to int type?
    k = 5
    position_top_k_mode, price_top_k_mode , dist_top_k_mode, eta_top_k_mode = -np.ones((n, k)), -np.ones((n, k)), -np.ones((n, k)), -np.ones((n, k))
    mode_texts = []
    max_num_mode = 0
    for i, [plan, hour, click_mode] in tqdm(enumerate(data[['plans', 'hour','click_mode']].values)):
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

            max_eta_per_dist[i], min_eta_per_dist[i], mean_eta_per_dist[i], \
            std_eta_per_dist[i] = np.nan, np.nan, np.nan, np.nan

            # max_price_per_dist[i], min_price_per_dist[i], mean_price_per_dist[i], \
            # std_price_per_dist[i] = np.nan, np.nan, np.nan, np.nan

            min_dist_mode[i] = -1
            max_dist_mode[i] = -1
            min_price_mode[i] = -1
            max_price_mode[i] = -1
            min_eta_mode[i] = -1
            max_eta_mode[i] = -1
            first_mode[i] = -1
            last_mode[i]= -1
            mode_texts.append('word_null')
        else:
            distance_list = []
            price_list = []
            eta_list = []
            mode_list = []
            eta_per_dist_list = []
            # price_per_dist_list = []

            for ind, tmp_dict in enumerate(cur_plan_list):
                distance_list.append(tmp_dict['distance'])
                if tmp_dict['price'] == '':
                    cal_price = price_dict['price_per_distance'][(tmp_dict['transport_mode'], hour)] * tmp_dict['distance']
                else:
                    cal_price = tmp_dict['price']
                price_list.append(cal_price)
                eta_list.append(tmp_dict['eta'])
                mode_list.append(tmp_dict['transport_mode'])
                eta_per_dist_list.append(tmp_dict['eta']/(tmp_dict['distance'] + 0.0001))
                # price_per_dist_list.append(cal_price/(tmp_dict['distance'] + 0.0001))
                # if ind < k:
                #     position_top_k_mode[i, ind] = tmp_dict['transport_mode']

            mode_texts.append(
                ' '.join(['word_{}'.format(mode) for mode in mode_list]))
            #drop duplicate index
            mode_list, distance_list, price_list, eta_list, eta_per_dist_list = drop_duplicate_mode(mode_list, \
                                            distance_list, price_list, eta_list, eta_per_dist_list)
            # This represents the case where no user clicks the plan
            # so mode 0 must be (one of) the available plan(s) for the user
            #########################
            # test the impact of this#
            # changes both mode_avail_list and mode_rank_list
            #########################
            # if click_mode == 0:
            #     mode_list.append(0)
            #####
            # by default, the last avail mode is  0 (user can always choose to not click any of the plan)
            # mode_list.append(0)
            max_num_mode = max(len(mode_list), max_num_mode)
            mode_list = np.array(mode_list, dtype='int')
            distance_list = np.array(distance_list)
            price_list = np.array(price_list)
            eta_list = np.array(eta_list)
            eta_per_dist_list = np.array(eta_per_dist_list)
            # The following assignment does not matter even if there are duplicates in mode_list
            mode_avail_list[i, mode_list] = 1
            for index, trans_mode in enumerate(mode_list):
                mode_rank_list[i, trans_mode] = 9-index

            distance_sort_idx = np.argsort(distance_list)
            price_sort_idx = np.argsort(price_list)
            eta_sort_idx = np.argsort(eta_list)
            eta_per_dist_idx = np.argsort(eta_per_dist_list)

            # mode_len = len(distance_sort_idx)
            # if mode_len <= k:
            #     dist_top_k_mode[i, :mode_len] = mode_list[distance_sort_idx]
            #     price_top_k_mode[i, :mode_len] = mode_list[price_sort_idx]
            #     eta_top_k_mode[i, :mode_len] = mode_list[eta_sort_idx]
            # else:
            #     dist_top_k_mode[i] = mode_list[distance_sort_idx[:k]]
            #     price_top_k_mode[i] = mode_list[price_sort_idx[:k]]
            #     eta_top_k_mode[i] = mode_list[eta_sort_idx[:k]]

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

            max_eta_per_dist[i] = eta_per_dist_list[eta_per_dist_idx[-1]]
            min_eta_per_dist[i] = eta_per_dist_list[eta_per_dist_idx[0]]
            mean_eta_per_dist[i] = np.mean(eta_per_dist_list)
            std_eta_per_dist[i] = np.std(eta_per_dist_list)

            # max_price_per_dist[i] = price_per_dist_list[price_per_dist_idx[-1]]
            # min_price_per_dist[i] = price_per_dist_list[price_per_dist_idx[0]]
            # mean_price_per_dist[i] = np.mean(price_per_dist_list)
            # std_price_per_dist[i] = np.std(price_per_dist_list)

            first_mode[i] = mode_list[0]
            last_mode[i]=mode_list[-1]
            max_dist_mode[i] = mode_list[distance_sort_idx[-1]]
            min_dist_mode[i] = mode_list[distance_sort_idx[0]]

            max_price_mode[i] = mode_list[price_sort_idx[-1]]
            min_price_mode[i] = mode_list[price_sort_idx[0]]

            max_eta_mode[i] = mode_list[eta_sort_idx[-1]]
            min_eta_mode[i] = mode_list[eta_sort_idx[0]]

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

    feature_data['max_eta_per_dist'] = max_eta_per_dist
    feature_data['min_eta_per_dist'] = min_eta_per_dist
    feature_data['mean_eta_per_dist'] = mean_eta_per_dist
    feature_data['std_eta_per_dist'] = std_eta_per_dist

    # feature_data['max_price_per_dist'] = max_price_per_dist
    # feature_data['min_price_per_dist'] = min_price_per_dist
    # feature_data['mean_price_per_dist'] = mean_price_per_dist
    # feature_data['std_price_per_dist'] = std_price_per_dist

    feature_data['max_dist_mode'] = max_dist_mode
    feature_data['min_dist_mode'] = min_dist_mode
    feature_data['max_price_mode'] = max_price_mode
    feature_data['min_price_mode'] = min_price_mode
    feature_data['max_eta_mode'] = max_eta_mode
    feature_data['min_eta_mode'] = min_eta_mode
    feature_data['first_mode'] = first_mode
    feature_data['last_mode'] = last_mode

    # for rank_type in ['rank', 'dist', 'price','eta']:
    #     for i, col in enumerate([rank_type+'_1_mode', rank_type+'_2_mode', rank_type+'_3_mode', \
    #                                 rank_type+'_4_mode', rank_type+'_5_mode']):
    #         if rank_type == 'rank':
    #             feature_data[col] = position_top_k_mode[:,i]
    #         elif rank_type == 'dist':
    #             feature_data[col] = dist_top_k_mode[:,i]
    #         elif rank_type == 'price':
    #             feature_data[col] = price_top_k_mode[:,i]
    #         else:
    #             feature_data[col] = eta_top_k_mode[:,i]

    mode_svd = gen_mode_context_feas(mode_texts)
    data = pd.concat([data, feature_data, mode_svd], axis=1)

    df_mode_avail = pd.DataFrame(mode_avail_list)
    df_mode_avail.columns = ['mode_avail_{}'.format(i) for i in range(12)]
    data = pd.concat([data, df_mode_avail], axis =1)

    data = data.drop(['plans'], axis=1)
    print('max num of mode per session is:', max_num_mode)
    return data, mode_avail_list, mode_rank_list

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

    # may try np.log1p()
    for col in transform_fea_list:
        data['sqr_'+col] = np.square(data[col])
        data['sqrt_'+col] = np.sqrt(data[col])
    return data

def gen_basic_profile_repre(data, profile_data):
    n_profile_enc = 30
    x = profile_data.drop(['pid', 'cluster_id'], axis=1).values
    svd = TruncatedSVD(n_components=n_profile_enc, random_state=2019)
    svd_x = svd.fit_transform(x)
    df_profile_repre = pd.DataFrame(svd_x)
    df_profile_repre.columns = ['profile_repre_{}'.format(i) for i in range(n_profile_enc)]
    df_profile_repre['pid'] = profile_data['pid'].values
    df_profile_repre['cluster_id'] = profile_data['cluster_id'].values
    # ave_repre = list(df_profile_repre.mean().values)
    # df_profile_repre['pid'] = profile_data['pid'].values
    # df_profile_repre = df_profile_repre.append(pd.DataFrame([ave_repre + [-2]], columns = df_profile_repre.columns))
    #pid =-1 corresponds to empty plan list; the click mode should be 0 or -1
    #pid =-2 corresponds to simply missing pid but with plan list; the click mode can be -1, 0, 1, ..., 11.
    data['pid']= np.where(data['pid'].isnull(), np.where(data['mode_rank_0']>0, -1, -2), data['pid'])
    data = data.merge(df_profile_repre, on='pid', how='left')
    return data

def gen_time_weather_feas(data):
    data['req_time'] = pd.to_datetime(data['req_time'])
    data['date'] = data['req_time'].dt.normalize()
    data['weekday'] = data['req_time'].dt.dayofweek
    data['hour'] = data['req_time'].dt.hour
    data['time_index']= ((data['req_time'].dt.hour * 60 + data['req_time'].dt.minute)//10).astype(int)

    df_weather = gen_weather_data()
    data = data.merge(df_weather, on ='date', how = 'left')
    data = data.drop(['req_time', 'date'], axis=1)
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

    # n_pid * 12 = (n_pid * n_encoding) *  (n_encoding * n_encoding) * (n_encoding * 12)
    x = df_mode_avail.loc[:, cols].values
    svd = TruncatedSVD(n_components=n_encoding, random_state=2019)
    # n_pid * n_encoding
    svd_x = svd.fit_transform(x)
    # (n_pid * n_encoding) * (n_encoding * 12)
    similarity_x = np.matmul(svd_x, svd.components_)
    df_profile_avail_mode_repre = pd.DataFrame(svd_x)
    df_similarity = pd.DataFrame(similarity_x)
    df_profile_avail_mode_repre.columns = ['profile_avail_mode_repre_{}'.format(i) for i in range(n_encoding)]
    df_similarity.columns = ['similarity_from_profile_avail_{}'.format(i) for i in range(12)]
    # df_mode_avail = pd.concat([df_mode_avail, df_profile_avail_mode_repre, df_similarity], axis=1)
    df_mode_avail = pd.concat([df_mode_avail, df_similarity], axis=1)
    # litez: The obtained df_similarity gives similar ranking among trans mode as that given by mode_avail_;
    # Thus drop the latter may save the model from redundency
    df_mode_avail = df_mode_avail.drop(cols, axis =1)
    data = data.merge(df_mode_avail, on = 'pid', how = 'left')
    return data

def gen_profile_embedding(data, profile_data, n_encoding = 10):
    '''based on the click_mode (of cluster_id) as well as the one-hot encoding (of pid)'''

    ##
    # get the percentage rather than counts???
    ##
    sel_data = (data.query('click_mode != -1')
                    .groupby(['cluster_id','click_mode']).size()
                    .reset_index()
                    )
    sel_data.columns = ['cluster_id',  'click_mode', 'frequency']

    sel_data_wide = pd.pivot_table(sel_data, index = 'cluster_id', columns = 'click_mode')
    sel_data_wide.columns = ['mode_' + str(int(item[1])) for item in sel_data_wide.columns.values]
    sel_data_wide = sel_data_wide.reset_index().fillna(0)

    profile_data = profile_data.merge(sel_data_wide, on = 'cluster_id', how = 'left').fillna(0).sort_values('cluster_id')
    # # x dim: (12+66=78) * pid
    x_T = profile_data.drop(['pid', 'cluster_id'], axis =1).values.T
    svd = TruncatedSVD(n_components=n_encoding, random_state=2019)
    # 78 * n_encoding
    svd_x_T = svd.fit_transform(x_T)
    # n_pid * n_encoding
    profile_embedding = svd.components_.T

    return profile_embedding, profile_data['pid'].values
    # return profile_embedding, sel_data_wide['cluster_id'].values

def gen_mode_embedding(mode_avail_list, n_encoding = 10):
    svd = TruncatedSVD(n_components=n_encoding, random_state=2019)
    svd_x = svd.fit_transform(mode_avail_list)
    # n_encoding * 12
    mode_embedding_tz = svd.components_
    return mode_embedding_tz

def gen_mode_embedding_2(data, profile_data, n_encoding = 10):
    '''use the profile_data rather than avail_mode'''
    sel_data = (data.query('click_mode != -1')
                    .groupby(['cluster_id','click_mode']).size()
                    .reset_index()
                    )
    sel_data.columns = ['cluster_id',  'click_mode', 'frequency']

    sel_data_wide = pd.pivot_table(sel_data, index = 'cluster_id', columns = 'click_mode')
    sel_data_wide = sel_data_wide.fillna(0)
    # # x dim: cluster_id * 12
    x = sel_data_wide.values
    svd = TruncatedSVD(n_components=n_encoding, random_state=2019)
    # pid * n_encoding
    svd_x = svd.fit_transform(x)
    #  n_encoding * 12
    mode_embedding = svd.components_
    return mode_embedding

def gen_profile_mode_similarity(data, profile_data, mode_avail_list):
    print('generating profile-mode similarity score...')
    n_encoding = 10

    profile_embedding, id_list = gen_profile_embedding(data, profile_data, n_encoding)
    mode_embedding = gen_mode_embedding(mode_avail_list, n_encoding)
    # mode_embedding = gen_mode_embedding_2(data, profile_data, n_encoding)
    score_matrix = np.matmul(profile_embedding, mode_embedding)
    df_score = pd.DataFrame(score_matrix, index = id_list).reset_index()
    print('df_score.shape:', df_score.shape)
    df_score.columns = ['pid'] + ['similarity_score_' + str(i) for i in range (12)]
    data = data.merge(df_score, on = 'pid', how = 'left')
    return data

def gen_sel_mode_by_cluster(data):
    n_encoding = 6
    sel_data = (data.query('click_mode != -1')
                    .groupby(['cluster_id','click_mode']).size()
                    .reset_index()
                    )
    sel_data.columns = ['cluster_id',  'click_mode', 'frequency']

    sel_data_wide = pd.pivot_table(sel_data, index = 'cluster_id', columns = 'click_mode')
    sel_data_wide = sel_data_wide.fillna(0)
    sel_data_wide['row_sum']= sel_data_wide.sum(axis =1)
    for col in sel_data_wide.columns[:-1]:
        sel_data_wide[col] = sel_data_wide[col]/(sel_data_wide['row_sum'] + 0.001)
    sel_data_wide = sel_data_wide.drop(['row_sum'], axis =1)
    # # x dim: cluster_id * 12
    x = sel_data_wide.values
    svd = TruncatedSVD(n_components=n_encoding, random_state=2019)
    # pid * n_encoding
    svd_x = svd.fit_transform(x)
    df_cluster = pd.DataFrame(svd_x)
    df_cluster.columns = ['click_mode_repre' + str(i) for i in range(n_encoding)]

    data = pd.concat([data, df_cluster], axis = 1)
    return data

def split_train_test_11_class(data):
    train_data = data[data['click_mode'] != -1]
    train_data = train_data.drop(['sid', 'pid'], axis=1)
    train_y = train_data['click_mode'].values
    train_x = train_data.drop(['click_mode'], axis=1)

    test_data_raw = data.query('click_mode == -1')
    test_data = test_data_raw.query('mode_rank_0 == 0')
    test_data_null = test_data_raw.query('mode_rank_0 > 0')
    submit = test_data[['sid']].copy()

    test_x = test_data.drop(['sid', 'pid', 'click_mode'], axis=1)
    return train_x, train_y, test_x, test_data_null, submit

def split_train_test_12_class(data):
    train_data = data[data['click_mode'] != -1]
    train_data = train_data.drop(['sid', 'pid'], axis=1)
    train_y = train_data['click_mode'].values
    train_x = train_data.drop(['click_mode'], axis=1)

    test_data = data.query('click_mode == -1')
    submit = test_data[['sid']].copy()
    test_x = test_data.drop(['sid', 'pid', 'click_mode'], axis=1)
    return train_x, train_y, test_x, submit

def get_train_test_feas_data():
    '''check a clever way of filling missing prices;
        and fillnas
        why are there so many sids with min price ==0?
    '''
    profile_data = read_profile_data()
    # data = merge_raw_data().loc[:5000,:]
    data = merge_raw_data()
    data = gen_od_feas(data)
    data = gen_time_weather_feas(data)
    price_dict = gen_average_price(data)
    data, mode_avail_list, mode_rank_list = gen_plan_feas(data, price_dict)
    data = feature_transform(data)
    data = gen_basic_profile_repre(data, profile_data)
    data = gen_profile_avail_mode_repre(data, mode_avail_list)
    # data = gen_profile_mode_similarity(data, profile_data, mode_avail_list)
    data = gen_sel_mode_by_cluster(data)
    # force type convertion
    cate_cols = ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
                 'min_price_mode', 'max_eta_mode', 'min_eta_mode', 'first_mode', 'last_mode','weekday', 'hour', 'weather',
                 'cate_o1', 'cate_o2','cate_d1','cate_d2'
                 ]
    for col in cate_cols:
        data[col] = data[col].astype(int)

    # data.to_csv("../output/merge_data.csv", index=False)
    print('shape of the dataframe:', data.shape, '\n')
    train_x, train_y, test_x, submit = split_train_test_12_class(data)
    return train_x, train_y, test_x, submit


if __name__ == '__main__':
    train_x, train_y, test_x, submit = get_train_test_feas_data()
    train_x.to_csv('../tmp/train_x.csv', index = False)
    df_train_y = pd.DataFrame({'train_y':train_y})
    df_train_y.to_csv('../tmp/train_y.csv', index = False)
    test_x.to_csv('../tmp/test_x.csv', index = False)
    submit.to_csv('../tmp/submit.csv', index = False)

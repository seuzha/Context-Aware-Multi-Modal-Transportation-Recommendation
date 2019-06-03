# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb

import gen_features

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from time import gmtime, strftime
import datetime

def read_data_files(file_path):
    train_x = pd.read_csv(file_path + 'train_x.csv')
    train_y = pd.read_csv(file_path + 'train_y.csv')

    #change df to np array
    train_y = np.array(list(train_y.to_dict()['train_y'].values()))
    test_x = pd.read_csv(file_path + 'test_x.csv')
    submit = pd.read_csv(file_path + 'submit.csv')

    #check if a few features are necessary
    drop_cols = ['time_index','cluster_id']
    train_x = train_x.drop(drop_cols, axis =1)
    test_x  = test_x.drop(drop_cols, axis =1)

    return train_x, train_y, test_x, submit

def gen_feature_importance(lgb_model):
    feature_importance = pd.DataFrame({'name':lgb_model.feature_name(), \
                                        'importance':lgb_model.feature_importance()}).sort_values(by='importance', ascending=False)
    feature_importance.to_csv('../tmp/feat_importance.csv', index=False)

def eval_f(y_pred, train_data):
    y_true = train_data.label
    y_pred = y_pred.reshape((12, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    score = f1_score(y_true, y_pred, average='weighted')
    return 'weighted-f1-score', score, True

def f1_decomposition(val_y, val_pred):
    precision, recall, F1, support = precision_recall_fscore_support(val_y, val_pred)
    weighted_F1 =  precision_recall_fscore_support(val_y, val_pred, average ='weighted')[2]
    df_eval = pd.DataFrame({'precision':precision, 'recall':recall,'F1':F1, 'support':support, 'weighted_F1':weighted_F1})
    return df_eval

def augment_minority_class(val_proba, val_y):

    val_score_list = []
    f1_decomposition_list=[]
    search_len = 20
    for weight1 in range(search_len):
        for weight2 in range(search_len):
            val_proba_tmp = val_proba.copy()
            weight1 = weight1/10.0 + 1
            weight2 = weight2/10.0 + 1
            val_proba_tmp[:,3]=val_proba_tmp[:,3]*weight1
            val_proba_tmp[:,4]=val_proba_tmp[:,4]*weight2
            val_pred_tmp = np.argmax(val_proba_tmp, axis=1)
            val_score = f1_score(val_y, val_pred_tmp, average='weighted')
            df_f1_decomposition = f1_decomposition(val_y, val_pred_tmp)
            val_score_list.append(val_score)
            f1_decomposition_list.append(df_f1_decomposition)
    max_index = np.argmax(np.array(val_score_list))
    print('weight1:', max_index//search_len, 'weight2:', max_index%search_len)
    print(f1_decomposition_list[max_index])

def regulize_proba_by_avail_mode(pred_proba, mode_avail):
    output_proba = np.multiply(pred_proba, mode_avail)
    return output_proba

def submit_result_12_class(submit, pred_test, model_name):
    now_time = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    submit['recommend_mode'] = pred_test
    submit.to_csv(
        '../submit/{}_result_{}.csv'.format(model_name, now_time), index=False)

def submit_result_11_class(submit, pred_test, test_data_null, model_name):
    now_time = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    submit['recommend_mode'] = pred_test
    test_data_null['recommend_mode'] = 0
    submit = submit.append(test_data_null.loc[:, ['sid', 'recommend_mode']], ignore_index = True)
    submit.to_csv(
        '../submit/{}_result_{}.csv'.format(model_name, now_time), index=False)

def output_for_nn(tr_x, tr_y, val_x, val_y, test_x):
    pd.DataFrame(tr_x).to_csv('../output/tr_x.csv', index =False)
    pd.DataFrame({'tr_y':tr_y}).to_csv('../output/tr_y.csv', index =False)
    pd.DataFrame(val_x).to_csv('../output/val_x.csv', index =False)
    pd.DataFrame({'val_y':val_y}).to_csv('../output/val_y.csv', index =False)
    pd.DataFrame(test_x).to_csv('../output/test_x.csv', index =False)

def train_lgb(train_x, train_y, test_x):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    avail_cols = ['mode_avail_{}'.format(i) for i in range(12)]
    test_avail_mode = test_x[avail_cols].values
    test_x = test_x.drop(avail_cols, axis = 1)

    lgb_paras = {
        'objective': 'multiclass',
        # 'metrics': 'multiclass',
        'learning_rate': 0.02,
        'num_leaves': 255,
        # 'max_depth':
        # 'min_data_in_leaf':

        # 'lambda_l1': 0.01,
        'lambda_l1': 1, # !!!original is 1
        'lambda_l2': 10,
        'num_class': 12,
        'seed': 2019,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
        'verbose':-1, # try to turn off the warnings
        'early_stopping_rounds':100,
    }
    cate_cols = ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
                 'min_price_mode', 'max_eta_mode', 'min_eta_mode', 'first_mode', 'last_mode','weekday', 'hour', 'weather',
                 'cate_o1', 'cate_o2','cate_d1','cate_d2'
                 ]
    # cate_cols = ['weekday', 'hour', 'weather']
    # for rank_type in ['rank', 'dist', 'price','eta']:
    #     cate_cols = cate_cols + [rank_type+'_1_mode', rank_type+'_2_mode', rank_type+'_3_mode', rank_type+'_4_mode', rank_type+'_5_mode']
    result_proba = []
    cv_count =0
    for tr_idx, val_idx in kfold.split(train_x, train_y):
        if cv_count<1:
            tr_x, tr_y, val_x, val_y = train_x.iloc[tr_idx], train_y[tr_idx], train_x.iloc[val_idx], train_y[val_idx]
            tr_avail_mode = tr_x[avail_cols].values
            val_avail_mode = val_x[avail_cols].values
            tr_x = tr_x.drop(avail_cols, axis =1)
            val_x = val_x.drop(avail_cols, axis =1)

            train_set = lgb.Dataset(tr_x, tr_y, categorical_feature=cate_cols)
            val_set = lgb.Dataset(val_x, val_y, categorical_feature=cate_cols)
            lgb_model = lgb.train(lgb_paras, train_set, valid_sets=[val_set], num_boost_round=1000, verbose_eval=50, feval=eval_f)
            # train_proba = lgb_model.predict(tr_x, num_iteration = lgb_model.best_iteration)
            print('generating val_proba', '\n')
            val_proba = lgb_model.predict(val_x, num_iteration=lgb_model.best_iteration)
            augment_minority_class(val_proba, val_y)
            # val_proba = regulize_proba_by_avail_mode(val_proba, val_avail_mode)
            val_pred = np.argmax(val_proba, axis=1)
            print('original f1:', f1_decomposition(val_y, val_pred), '\n')
            print('generating result_proba', '\n')
            result_proba = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration)
            # re-weight for class 3
            # result_proba[:,3] = 2.4 * result_proba[:,3]
            # result_proba = regulize_proba_by_avail_mode(result_proba, test_avail_mode)
            # add a feed-forward nn
            # output_for_nn(train_proba, tr_y, val_proba, val_y, result_proba)

            gen_feature_importance(lgb_model)
            cv_count+=1

    result_proba[:,3] = 2.4 * result_proba[:,3]
    pred_test = np.argmax(result_proba, axis=1)
    return pred_test

if __name__ == '__main__':
    train_x, train_y, test_x, submit = gen_features.get_train_test_feas_data()
    train_x = train_x.drop(['time_index', 'cluster_id'], axis =1)
    test_x = test_x.drop(['time_index', 'cluster_id'], axis =1)
    # train_x, train_y, test_x, submit = read_data_files('../tmp/')
    pred_test = train_lgb(train_x, train_y, test_x)
    submit_result_12_class(submit, pred_test, 'lgb')
    # submit_result_11_class(submit, pred_test, test_data_null, 'lgb')

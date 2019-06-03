from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
import datetime
from tqdm import tqdm


def f1_decomposition(val_y, val_pred):
    precision, recall, F1, support = precision_recall_fscore_support(val_y, val_pred)
    weighted_F1 =  precision_recall_fscore_support(val_y, val_pred, average ='weighted')[2]
    df_eval = pd.DataFrame({'precision':precision, 'recall':recall,'F1':F1, 'support':support, 'weighted_F1':weighted_F1})
    return df_eval

# def augment_minority_class(val_y, val_proba):
#
#     val_score_list = []
#     f1_decomposition_list=[]
#     search_len = 20
#     for weight1 in range(search_len):
#         for weight2 in range(search_len):
#             val_proba_tmp = val_proba.copy()
#             weight1 = weight1/10.0 + 1
#             weight2 = weight2/10.0 + 1
#             val_proba_tmp[:,3]=val_proba_tmp[:,3]*weight1
#             val_proba_tmp[:,4]=val_proba_tmp[:,4]*weight2
#             val_pred_tmp = np.argmax(val_proba_tmp, axis=1)
#             val_score = f1_score(val_y, val_pred_tmp, average='weighted')
#             df_f1_decomposition = f1_decomposition(val_y, val_pred_tmp)
#             val_score_list.append(val_score)
#             f1_decomposition_list.append(df_f1_decomposition)
#     max_index = np.argmax(np.array(val_score_list))
#     print('weight1:', max_index//search_len, 'weight2:', max_index%search_len)
#     print(f1_decomposition_list[max_index])


def augment_minority_class(val_y, val_proba):

    val_score_list = []
    f1_decomposition_list=[]
    search_len = 20
    for weight0 in range(search_len):
        val_proba_tmp = val_proba.copy()
        weight0 = weight0/10.0 + 1
        val_proba_tmp[:,0]=val_proba_tmp[:,0]*weight0
        val_proba_tmp[:,3]=val_proba_tmp[:,3]*2.5
        val_pred_tmp = np.argmax(val_proba_tmp, axis=1)
        val_score = f1_score(val_y, val_pred_tmp, average='weighted')
        df_f1_decomposition = f1_decomposition(val_y, val_pred_tmp)
        val_score_list.append(val_score)
        f1_decomposition_list.append(df_f1_decomposition)
    max_index = np.argmax(np.array(val_score_list))
    print('weight0:', max_index)
    print(val_score_list)
    print(f1_decomposition_list[max_index])

def augment_minority_class_(val_y, val_proba):

    val_score_list = []
    f1_decomposition_list=[]
    search_len = 20
    for weight0 in tqdm(range(search_len)):
        for weight1 in range(search_len):
            for weight2 in range(search_len):
                val_proba_tmp = val_proba.copy()
                weight0 = weight0/10.0 + 1
                weight1 = weight1/10.0 + 1
                weight2 = weight2/10.0 + 1
                val_proba_tmp[:,0]=val_proba_tmp[:,0]*weight0
                val_proba_tmp[:,3]=val_proba_tmp[:,3]*weight1
                val_proba_tmp[:,4]=val_proba_tmp[:,4]*weight2
                val_pred_tmp = np.argmax(val_proba_tmp, axis=1)
                val_score = f1_score(val_y, val_pred_tmp, average='weighted')
                df_f1_decomposition = f1_decomposition(val_y, val_pred_tmp)
                val_score_list.append(val_score)
                f1_decomposition_list.append(df_f1_decomposition)
    max_index = np.argmax(np.array(val_score_list))
    max_index_copy = max_index
    index_list = []
    while max_index>0:
        index_list.append(max_index%search_len)
        max_index = max_index//search_len

    print('weight0:', index_list.pop(),'weight1:', index_list.pop(), 'weight2:', index_list.pop())
    print(f1_decomposition_list[max_index_copy])

def train(tr_x, tr_y, val_x, val_y, classifier_name):

    if classifier_name == 'logistical':
        # clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='lbfgs', class_weight='balanced').fit(tr_x, tr_y)
        clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='lbfgs', class_weight = {3:0.3}).fit(tr_x, tr_y)
        # clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='lbfgs', \
        #                             class_weight= {0:0.093, 1: 0.141, 2: 0.273, 3:0.049, 4:0.025, 5:0.095, 6:0.024,\
        #                                                 7:0.156, 8:0.004, 9:0.098, 10:0.03, 11:0.012}).fit(tr_x, tr_y)

    elif classifier_name == 'nn':
        clf = MLPClassifier(hidden_layer_sizes=(1, )).fit(tr_x, tr_y)
    elif classifier_name == 'rf':
        clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)).fit(tr_x, tr_y)
    result = clf.predict(val_x)
    print(f1_decomposition(val_y, result))
    return clf


def read_files():
    tr_x = pd.read_csv('../output/tr_x.csv')
    tr_y = pd.read_csv('../output/tr_y.csv')
    val_x = pd.read_csv('../output/val_x.csv')
    val_y = pd.read_csv('../output/val_y.csv')
    test_x = pd.read_csv('../output/test_x.csv')
    submit = pd.read_csv('../tmp/submit.csv')
    return tr_x, tr_y, val_x, val_y, test_x, submit


def submit_result(submit, pred_test, model_name):
    now_time = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    submit['recommend_mode'] = pred_test
    submit.to_csv(
        '../output/{}_result_{}.csv'.format(model_name, now_time), index=False)

def preprocess_data():
    tr_x, tr_y, val_x, val_y, test_x, submit = read_files()
    print('before:', f1_decomposition(val_y['val_y'].values, np.argmax(val_x.values, axis=1)), '\n')
    augment_minority_class(val_y['val_y'].values, val_x.values)
    # augment_minority_class_(val_y['val_y'].values, val_x.values)

    tr_x = tr_x.values
    tr_y = tr_y['tr_y'].values
    val_x = val_x.values
    val_y = val_y['val_y'].values
    test_x = test_x.values

    classifier_name = 'logistical'
    # classifier_name = 'rf'
    # classifier_name = 'nn'
    # clf = train(tr_x, tr_y, val_x, val_y, classifier_name)
    # pred_test = clf.predict(test_x).astype(int)
    # submit_result(submit, pred_test, classifier_name)

if __name__ == '__main__':
    preprocess_data()

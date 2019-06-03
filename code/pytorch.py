import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import datetime
import gen_features_torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def read_data_files(file_path):
    data = pd.read_csv(file_path + 'data_long.csv')
    return data

def gen_batch_data(data):
    cols = list(data.columns)
    del_cols_index = [cols.index(col) for col in ['sid', 'pid', 'click_mode']]
    sel_cols_index = list(range(len(cols)))
    for item in del_cols_index:
        sel_cols_index.pop(item)

    grouped = data.groupby('sid')
    batch_feas_list = []
    batch_click_mode_list = []
    for i, (group_id, group) in tqdm(enumerate(grouped)):
        grouped_values = group.values
        batch_click_mode = grouped_values[:,del_cols_index[-1]][0]
        batch_feas = torch.tensor(grouped_values[:, sel_cols_index]).type(torch.FloatTensor)
        batch_feas_list.append(batch_feas)
        batch_click_mode_list.append(batch_click_mode)
    batch_click_mode_list = torch.tensor(batch_click_mode_list).type(torch.LongTensor)
    # first paddle
    # check why size does not add up to 10000???
    print('list_len:', len(batch_click_mode_list))
    return batch_feas_list, batch_click_mode_list

def split_train_test_12_class(file_path):
    # data = read_data_files(file_path)
    data = gen_features_torch.merge_data()
    train_data = data[data['click_mode'] != -1]
    #train-val split
    val_data = train_data.iloc[:int(0.2 * train_data.shape[0]), :]
    train_data = train_data.iloc[int(0.2 * train_data.shape[0]):, :]

    train_x, train_y = gen_batch_data(train_data)
    val_x, val_y = gen_batch_data(val_data)

    test_data = data.query('click_mode == -1')
    submit = test_data[['sid']].copy()

    test_x, _ = gen_batch_data(test_data)
    return train_x, train_y, val_x, val_y, test_x, submit

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

def submit_result_12_class(submit, pred_test, model_name):
    now_time = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    submit['recommend_mode'] = pred_test
    submit.to_csv(
        '../submit/{}_result_{}.csv'.format(model_name, now_time), index=False)

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # batch_first=True will affect the input shape to self.rnn()

        self.rnn = nn.RNN(input_dim, 64, layer_dim, batch_first=True,
                          nonlinearity='relu')
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        out, hn = self.rnn(x)
        #hn[-1] is of (batch_size, hidden_dim)
        # hn_ = self.dropout(hn[-1])
        out = self.fc1(hn[-1])
        out = self.fc2(out)
        # out = self.dropout(out)
        return out

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout = 0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # may want to initialize h0 and c0
        out, (hn,cn) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

def train_test(train_x, train_y, val_x, val_y, test_x):
    '''train_x is a list of two-dim tensor
    train_y is an one-dim tensor
    '''
    input_dim = train_x[0].size()[1]
    hidden_dim = 10
    output_dim = 12
    layer_dim = 1 #used in RNN
    batch_size = 100

    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
    # model = LSTM(input_dim, hidden_dim, layer_dim, output_dim)

    # Cross Entropy Loss
    error = nn.CrossEntropyLoss()
    # error = nn.CrossEntropyLoss(weight = torch.tensor(np.array([1,1,6,1,1,1,1,1,1,1,1,1])).type(torch.FloatTensor))

    #learning_rate = 1 for LSTM
    learning_rate = 0.000001 # for RNN with num_epochs 500 seems to have 0.29 F1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


    num_epochs = 50
    pred_test_list = []
    score_list = []
    print('# of iters:',(len(train_x)/batch_size),'\n')
    for epoch in range(num_epochs):
        for i in range(len(train_x)//batch_size+1):
            if i < len(train_x)//batch_size:
                feas = train_x[i*batch_size:(i+1)*batch_size]
                labels = train_y[i*batch_size:(i+1)*batch_size]
            else:
                feas = train_x[i*batch_size:]
                labels = train_y[i*batch_size:]
            # feas = feas.view(-1, 12, 1) #needed for RNN
            lens = list(map(len, feas))
            padded_feas = pad_sequence(feas, batch_first=True)
            packed_feas = pack_padded_sequence(padded_feas, lens, batch_first=True, enforce_sorted=False)
            optimizer.zero_grad()
            outputs = model(packed_feas)
            loss = error(outputs, labels)
            if i%100==0:
                print('loss at {0} is {1}'.format(i, loss.item()))
            loss.backward()
            optimizer.step()

            if i ==0:
                train_outputs = outputs
            else:
                train_outputs = torch.cat((train_outputs, outputs), dim = 0)
            i+=1

        with torch.no_grad():
            for i in range(len(val_x)//batch_size+1):
                if i < len(val_x)//batch_size:
                    val_feas = val_x[i*batch_size:(i+1)*batch_size]
                    val_labels = val_y[i*batch_size:(i+1)*batch_size]
                else:
                    val_feas = val_x[i*batch_size:]
                    val_labels = val_y[i*batch_size:]
                val_lens = list(map(len, val_feas))
                padded_val_feas = pad_sequence(val_feas, batch_first=True)
                packed_val_feas = pack_padded_sequence(padded_val_feas, val_lens, batch_first=True, enforce_sorted=False)
                val_outputs = model(packed_val_feas)
                predicted = torch.max(val_outputs.data, 1)[1]
                if i ==0:
                    val_result = predicted
                    #for evaluation purposes
                    val_outputs_long = val_outputs
                else:
                    val_result = torch.cat((val_result, predicted), dim = 0)
                    val_outputs_long = torch.cat((val_outputs_long, val_outputs), dim = 0)
    #     pred_test = predict(model, test_loader)
    #     pred_test_list.append(pred_test)
        score = precision_recall_fscore_support(val_y, val_result, average ='weighted')[2]
        # print('epoch:', epoch, 'f1-score:', score)
        score_list.append(score)
    max_index = np.argmax(np.array(score_list))
    print('max_index', max_index)
    print(f1_decomposition(val_y, val_result))
    val_outputs_long = pd.DataFrame(val_outputs_long.numpy())
    val_outputs_long['click_mode']=val_y
    val_outputs_long['predicted']=val_result
    val_outputs_long.to_csv('../output/val_outputs_long.csv')
    train_outputs_long = pd.DataFrame(train_outputs.detach().numpy())
    train_outputs_long['predicted']=np.argmax(train_outputs_long.values, axis=1)
    train_outputs_long['click_mode']=train_y
    train_outputs_long.to_csv('../output/train_outputs_long.csv')
    # pred_test = pred_test_list[max_index]
    # return pred_test

if __name__ == '__main__':
    train_x, train_y, val_x, val_y, test_x, submit = split_train_test_12_class('../input_torch/')
    train_test(train_x, train_y, val_x, val_y, test_x)
    # submit_result_12_class(submit, pred_test, 'pytorch')

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
import datetime
from sklearn.model_selection import StratifiedKFold

class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.sigmoid(out)

        return out

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        # Linear part
        self.linear = nn.Linear(input_dim, output_dim)
        # There should be logistic function right?
        # However logistic function in pytorch is in loss function
        # So actually we do not forget to put it, it is only at next parts

    def forward(self, x):
        out = self.linear(x)
        return out

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # batch_first=True will affect the input shape to self.rnn()
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True,
                          nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # One time step
        out, hn = self.rnn(x, h0)

        #as batch_first is set to true, out: (batch, seq, feature)
        #out[:,-1,:] reduce the dimension by 1
        out = self.fc(out[:, -1, :])
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
        out = self.fc(out[:, -1, :])
        return out

def f1_decomposition(val_y, val_pred):
    precision, recall, F1, support = precision_recall_fscore_support(val_y, val_pred)
    weighted_F1 =  precision_recall_fscore_support(val_y, val_pred, average ='weighted')[2]
    df_eval = pd.DataFrame({'precision':precision, 'recall':recall,'F1':F1, 'support':support, 'weighted_F1':weighted_F1})
    return df_eval

def submit_result(submit, pred_test, model_name='logistic_pytorch'):
    now_time = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    submit['recommend_mode'] = pred_test
    submit.to_csv(
        '../output/{}_result_{}.csv'.format(model_name, now_time), index=False)

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

def train_test(train_loader, val_loader, test_loader, val_y):
    input_dim = 12
    # input_dim = 1 #used in RNN
    hidden_dim = 20
    output_dim = 12
    layer_dim = 1 #used in RNN

    # model = ANNModel(input_dim, hidden_dim, output_dim)
    model = LogisticRegressionModel(input_dim, output_dim)
    # model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
    # model = LSTM(input_dim, hidden_dim, layer_dim, output_dim)

    # Cross Entropy Loss
    error = nn.CrossEntropyLoss()
    # error = nn.CrossEntropyLoss(weight = torch.ones(12))

    learning_rate = 0.05
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    num_epochs = 20
    pred_test_list = []
    score_list = []
    for epoch in range(num_epochs):
        for i, (feas, labels) in tqdm(enumerate(train_loader)):
            # if i==0:
            #     print(feas.size())
            #     print(labels.size())
            # feas = feas.view(-1, 12, 1) #needed for RNN
            optimizer.zero_grad()
            outputs = model(feas)
            loss = error(outputs, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for i, (feas_val, labels_val) in enumerate(val_loader):
                # no need to define Variable() here
                val_ = feas_val
                # val_ = feas_val.view(-1, 12, 1) #needed for RNN
                val_outputs = model(val_)
                predicted = torch.max(val_outputs.data, 1)[1]
                if i ==0:
                    val_result = predicted
                    val_outputs_long = val_outputs
                else:
                    val_result = torch.cat((val_result, predicted), dim = 0)
                    val_outputs_long = torch.cat((val_outputs_long, val_outputs), dim = 0)
        pred_test = predict(model, test_loader)
        pred_test_list.append(pred_test)
        score = precision_recall_fscore_support(val_y, val_result, average ='weighted')[2]
        print('epoch:', epoch, 'f1-score:', score)
        score_list.append(score)
    max_index = np.argmax(np.array(score_list))
    print('max_index', max_index)
    print('best_val:', f1_decomposition(val_y, val_result))
    pred_test = pred_test_list[max_index]
    return pred_test

def predict(model, test_loader):
    with torch.no_grad():
        for i, test_ in enumerate(test_loader):
            # no need to define Variable() here
            test_outputs = model(test_[0])
            # test_outputs = model(test_[0].view(-1, 12, 1)) # needed for RNN
            predicted = torch.max(test_outputs.data, 1)[1]
            if i ==0:
                pred_test = predicted
                # print('check what test_ looks like:', test_)
            else:
                pred_test = torch.cat((pred_test, predicted), dim = 0)

    return pred_test

def read_files():
    tr_x = pd.read_csv('../output/tr_x.csv')
    tr_y = pd.read_csv('../output/tr_y.csv')
    val_x = pd.read_csv('../output/val_x.csv')
    val_y = pd.read_csv('../output/val_y.csv')
    test_x = pd.read_csv('../output/test_x.csv')
    submit = pd.read_csv('../tmp/submit.csv')
    df_val = pd.concat([val_x, val_y], axis =1)
    return tr_x, tr_y, val_x, val_y, test_x, submit, df_val

def read_files_tmp():
    train_x = pd.read_csv('../tmp/train_x.csv')
    train_y = pd.read_csv('../tmp/train_y.csv')
    train_y = np.array(list(train_y.to_dict()['train_y'].values()))
    test_x = pd.read_csv('../tmp/test_x.csv')

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    avail_cols = ['mode_avail_{}'.format(i) for i in range(12)]
    cv_count = 0
    for tr_idx, val_idx in kfold.split(train_x, train_y):
        if cv_count<1:
            tr_x, tr_y, val_x, val_y = train_x.iloc[tr_idx], train_y[tr_idx], train_x.iloc[val_idx], train_y[val_idx]
            tr_avail_mode = tr_x[avail_cols].values
            val_avail_mode = val_x[avail_cols].values
            cv_count +=1

    test_avail_mode = test_x[avail_cols].values

    return tr_avail_mode, val_avail_mode, test_avail_mode

def preprocess_data():
    tr_avail_mode, val_avail_mode, test_avail_mode = read_files_tmp()
    tr_x, tr_y, val_x, val_y, test_x, submit, df_val = read_files()
    # tr_x.iloc[:, 3] = tr_x.iloc[:, 3] * 2.4
    # val_x.iloc[:, 3] = val_x.iloc[:, 3] * 2.4
    # test_x.iloc[:, 3] = test_x.iloc[:, 3] * 2.4

    val_x = np.multiply(val_x, val_avail_mode)
    test_x = np.multiply(test_x, test_avail_mode)
    # val_after = pd.DataFrame(np.multiply(val_x, val_avail_mode))
    # val_original = pd.DataFrame(val_x)
    # val_avail_mode = pd.DataFrame(val_avail_mode)
    # test = pd.concat([val_original, val_avail_mode, val_after], axis=1)
    # test['mode']=val_y['val_y'].values
    # test['before_pred_mode']=np.argmax(val_original.values, axis=1)
    # test['after_pred_mode']=np.argmax(val_after.values, axis =1)
    # test.to_csv('../output/test.csv')

    print('before:', f1_decomposition(val_y['val_y'].values, np.argmax(val_x.values, axis=1)), '\n')


    augment_minority_class(val_x.values, val_y['val_y'].values)
    test_proba = test_x.values
    test_proba[:, 4]=test_proba[:,4]*2.2
    pred_test = np.argmax(test_proba, axis =1)
    # submit_result(submit, pred_test)

    # tr_x = torch.from_numpy(tr_x.values).type(torch.FloatTensor)
    # tr_y = torch.from_numpy(tr_y['tr_y'].values).type(torch.LongTensor)
    # val_x = torch.from_numpy(val_x.values).type(torch.FloatTensor)
    # val_y = torch.from_numpy(val_y['val_y'].values).type(torch.LongTensor)
    # test_x = torch.from_numpy(test_x.values).type(torch.FloatTensor)
    # print(tr_x.size(), tr_y.size(),val_x.size(),val_y.size(), test_x.size())
    #
    #
    # tr = torch.utils.data.TensorDataset(tr_x, tr_y)
    # val = torch.utils.data.TensorDataset(val_x, val_y)
    # test = torch.utils.data.TensorDataset(test_x)
    #
    # batch_size = 100
    # train_loader = torch.utils.data.DataLoader(tr, batch_size = batch_size, shuffle = False)
    # val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
    # test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

    # pred_test = train_test(train_loader, val_loader, test_loader, val_y)
    # submit_result(submit, pred_test)
if __name__ == '__main__':
    preprocess_data()

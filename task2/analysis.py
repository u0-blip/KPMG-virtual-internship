# In[1]: import libraries
import sys
sys.path.append('../')
    
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from importlib import reload
from task2 import helper
from task2 import get_RFM
from task2 import plot_coor
reload(helper)
reload(get_RFM)
reload(plot_coor)

import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from datetime import datetime, timedelta,date
from sklearn.metrics import classification_report,confusion_matrix
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import yaml
from time import time



# In[2]: Import data

customer = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name="CustomerDemographic", header=1)
new_customer = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name="NewCustomerList", header=1)
transaction = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name="Transactions", header=1)
address = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name="CustomerAddress", header=1)



# In[3]: calculating the RFM values

tx_user = get_RFM.get_RFM(transaction, False)



# In[4]: Plotting cluster

plot_data = [
    go.Histogram(
        x=tx_user['Recency']
    )
]

plot_layout = go.Layout(
        title='Recency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


plot_data = [
    go.Histogram(
        x=tx_user['Frequency']
    )
]

plot_layout = go.Layout(
        title='Frequency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


plot_data = [
    go.Histogram(
        x=tx_user['profit']
    )
]

plot_layout = go.Layout(
        title='Monetary Value'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)



# In[5]: plotting segmentation
tx_graph = tx_user

plot_coor.plot_coor(tx_graph, 'Frequency', 'profit','seg_cluster')
plot_coor.plot_coor(tx_graph, 'Recency', 'Frequency','seg_cluster')
plot_coor.plot_coor(tx_graph, 'profit', 'Recency','seg_cluster')

# In[6] regression model_t

with open('param.yaml', 'rb') as f:
    params = yaml.load(f)

train = int(params['train'])

def get_dummies(df, dummies, keep=None):
    df = pd.get_dummies(df, columns=dummies, prefix=[dummy + '_dummy' for dummy in dummies])
    for col in df.columns:
        if keep is not None:
            keep.append('_dummy_')
            keep.append('customer_id')
        else:
            keep = ['_dummy_', 'customer_id']
        kept = False
        for k in keep:
            if k in col:
                kept = True
                break
        if not kept:
            df.drop(col, axis=1, inplace=True)
    return df


tx_3m = transaction[(transaction.transaction_date >= pd.to_datetime('2017-1-1')) & (transaction.transaction_date < pd.to_datetime('2017-6-1'))].reset_index(drop=True)
tx_user_3m = get_RFM.get_RFM(tx_3m, back_in_time = True)

tx_6m = transaction[(transaction.transaction_date >= pd.to_datetime('2017-01-01')) & (transaction.transaction_date <= pd.to_datetime('2017-12-30'))].reset_index(drop=True)
tx_user_6m = get_RFM.get_RFM(tx_6m, back_in_time = False)

tx_user_6m.rename(columns={'profit':'12 mth profit'}, inplace=True)

# plot_data = [
#     go.Histogram(
#         x=tx_user_6m['OverallScore1']
#     )
# ]

# plot_layout = go.Layout(
#         title='6m OverallScore1'
#     )
# fig = go.Figure(data=plot_data, layout=plot_layout)
# pyoff.iplot(fig)

# tx_user_6m = tx_user_6m[tx_user_6m['customer_id'].isin(tx_user_3m['customer_id'])]
# tx_merge = pd.merge(tx_user_3m, tx_user_6m[['customer_id', '12 mth profit']], on='customer_id', how='left')

tx_merge = tx_user_6m
tx_merge = tx_merge.fillna(0)
print(tx_merge.groupby('seg_cluster')['12 mth profit'].mean())
tx_graph = tx_merge

# plot_coor.plot_coor(tx_graph, 'OverallScore1', '12 mth profit', 'seg_cluster')


# tx_merge = tx_merge[tx_merge['12 mth profit']<tx_merge['12 mth profit'].quantile(0.99)]


acc = []

kmeans = KMeans(n_clusters=8)
kmeans.fit(tx_merge[['12 mth profit']])
tx_merge['LTVCluster'] = kmeans.predict(tx_merge[['12 mth profit']])
tx_merge = helper.order_cluster('LTVCluster', '12 mth profit',tx_merge,True)
tx_cluster = tx_merge.copy()

y = tx_cluster[['customer_id', 'LTVCluster', 'OverallScore1', '12 mth profit']]

tx_class = tx_cluster
# dummies = ['profitCluster', 'RecencyCluster', 'FrequencyCluster', 'seg_cluster']
# tx_class = get_dummies(tx_cluster, dummies)


if not train:
    customer_clean = helper.clean_gender(new_customer)
else:
    customer_clean = helper.clean_gender(customer)

def conv_int(val):
    if type(val) != int:
        val = int(val)
    return val
def conv_float(val):
    if type(val) != float:
        val = float(val)
    return val

def prep_input(customer_clean, address):
    customer_clean = pd.merge(customer_clean, address, on='customer_id')
    customer_clean['postcode'] = customer_clean['postcode'].apply(conv_int)
    customer_clean['property_valuation'] = customer_clean['property_valuation'].apply(conv_int)

    customer_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    customer_clean.fillna(0, inplace=True)
    customer_clean = helper.get_age(customer_clean)

    cluster = {
        'name': ['tenure', 'Age', 'past_3_years_bike_related_purchases'],
        'num': [10, 13, 20],
        'min': [2, 5, 5],
    }
    for name, num, min in zip(*cluster.values()):
        customer_clean, num = get_RFM.get_clust(customer_clean, name, num, min_cluster = min, mode='auto')
        print('optimal '+name+' cluster: ' + str(num))

    dummies = ['gender', 'wealth_segment', 'owns_car', 'deceased_indicator', 'postcode', 'state', 'property_valuation']
    
    dummies += [name + 'Cluster' for name in cluster['name']]

    customer_clean = get_dummies(customer_clean, dummies)
    return customer_clean

customer_clean = prep_input(customer_clean, address)

if train:
    X_demo = customer_clean[customer_clean['customer_id'].isin(y['customer_id'])]

    # corr_matrix = pd.merge(customer_clean, y, on='customer_id').corr()
    # print('cooralation matrix')
    # print(corr_matrix['LTVCluster'].sort_values(ascending=False).to_string())
    y = y[y['customer_id'].isin(X_demo['customer_id'])]

    model_t = 'regressor'


    # X_demo = pd.merge(X_demo, tx_class[tx_class['customer_id'].isin(X_demo['customer_id'])], on='customer_id')
    y_col = y.columns

    X_demo = pd.merge(X_demo, y, on='customer_id')

    X_demo = X_demo.sample(frac=1).reset_index(drop=True)
    y = X_demo[y_col]
    ids = X_demo['customer_id']

    X_demo.drop(columns=y_col, inplace=True)

    if model_t == 'classifier':
        y = y['LTVCluster']
    elif model_t == 'regressor':
        y = y['12 mth profit']

    X_train, X_test, y_train, y_test = train_test_split(X_demo, y, test_size=0.2, random_state=56)

else:
    X_demo = customer_clean

## In[10]:
# if train:
#     prev = time()
#     if model_t == 'classifier':
#         ltv_xgb_model_t = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1,objective= 'multi:softprob',n_jobs=-1, tree_method='gpu_hist', gpu_id=0).fit(X_train, y_train)
#     elif model_t == 'regressor':
#         ltv_xgb_model_t = xgb.XGBRegressor(max_depth=5, n_estimators=900, learning_rate=0.1,objective='reg:squarederror',n_jobs=-1, tree_method='gpu_hist', gpu_id=0).fit(X_train, y_train)
#     after = time()
#     print('elapsed: ' + str(after - prev))

# if model_t == 'classifier':
#     print('Accuracy of XGB classifier on training set: {:.2f}'
#         .format(ltv_xgb_model_t.score(X_train, y_train)))
#     print('Accuracy of XGB classifier on test set: {:.2f}'
#         .format(ltv_xgb_model_t.score(X_test[X_train.columns], y_test)))
# elif model_t == 'regressor':
#     _y_train = ltv_xgb_model_t.predict(X_train)
#     _y_test = ltv_xgb_model_t.predict(X_test)
#     train_mse = np.sqrt(np.mean(pow(y_train -_y_train ,2)))
#     test_mse = np.sqrt(np.mean(pow(y_test - _y_test, 2)))
#     print('STD = ' + str(np.std(y_train)))
#     print('MSE of XGB regressor on training set: {:.2f}'
#         .format(train_mse))
#     print('MSE of XGB regressor on test set: {:.2f}'
#         .format(test_mse))

# acc.append([ltv_xgb_model_t.score(X_train, y_train), ltv_xgb_model_t.score(X_test[X_train.columns], y_test)])

# print('customer percentage sector: ' + str(tx_class.groupby('LTVCluster').customer_id.count()/tx_class.customer_id.count()))


# print(classification_report(y_test, y_pred))

# In[11] create simple torch classifier
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.3):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return x

def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        output = model(images.cuda())
        test_loss += criterion(output, labels.cuda()).cpu().item()

        ## Calculating the accuracy 
        # model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output.cpu())
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


def train(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every=40, validate_every=120):
    train_loss = []
    test_loss = []
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1
            
            optimizer.zero_grad()
            
            output = model(images.cuda())
            loss = criterion(output.cuda(), labels.cuda())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.cpu().item()

            if steps % print_every == 0:
                # model in inference mode, dropout is off
                model.eval()
                
                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3e}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3e}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3e}".format(accuracy/len(testloader)))
                train_loss.append(running_loss)
                test_loss.append(test_loss)

                running_loss = 0
                
                # Make sure dropout and grads are on for training
                model.train()
    return train_loss, test_loss

batch_size = 64

learning_rate = 0.0002
model = Network(len(X_train.columns), 1, [400, 300, 200])
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if model_t == 'regressor':
    criterion = nn.MSELoss()
elif model_t == 'classifier':
    criterion = nn.CrossEntropyLoss()
# make sure the SHUFFLE your training data

data = TensorDataset(torch.from_numpy(np.asarray(X_train)).float(), torch.from_numpy(np.asarray(y_train)).float())
train_loader = DataLoader(data, shuffle=False, batch_size=batch_size)

data = TensorDataset(torch.from_numpy(np.asarray(X_test)).float(), torch.from_numpy(np.asarray(y_test)).float())
test_loader = DataLoader(data, shuffle=False, batch_size=batch_size)

train_loss, test_loss = train(model, train_loader, test_loader, criterion, optimizer, epochs=80, print_every=40, validate_every=120)


# In[10] getting the importance
from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(10,8))
plot_importance(ltv_xgb_model, ax=ax, max_num_features=30)

# %% write recommandations to excel
y_pred = ltv_xgb_model.predict(X_demo)
num_customer_rec = 1000
highest_value = np.argsort(-y_pred)
recommand_customer = ids.iloc[highest_value[:num_customer_rec]]
recommand_customer = customer[customer['customer_id'].isin(recommand_customer)]
with pd.ExcelWriter('KPMG_VI_New_raw_data_update_final.xlsx', mode='a') as f:
    recommand_customer.to_excel(f, 'recommandation')

# %%

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from task2 import helper
from importlib import reload
reload(helper)
from task2.helper import *
import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from datetime import datetime, timedelta,date
from sklearn.metrics import classification_report,confusion_matrix
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import yaml
from sklearn.metrics import silhouette_score

with open('C:\\Users\\peter\\source\\KPMG-virtual-internship\\task2\\param.yaml', 'rb') as f:
    params = yaml.load(f)

train = int(params['train'])
cluster_info = dict()


def get_clust(data, field, num_cluster, order=True, min_cluster=2,  mode='manual'):
    global cluster_info
    if train:
        if mode == 'auto':
            # mean = data[field].mean()
            # std = data[field].std()

            # data = data[data[field] < mean + 3*std]
            sil = []
            kmax = num_cluster

            # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
            for k in range(min_cluster, kmax+1):
                kmeans = KMeans(n_clusters = k).fit(data[[field]])
                labels = kmeans.labels_
                sil.append(silhouette_score(data[[field]], labels, metric = 'euclidean'))
            num_cluster = np.argmax(np.array(sil)) + min_cluster

        kmeans = KMeans(n_clusters=num_cluster)
        kmeans.fit(data[[field]])
        cluster_info[field] = kmeans
    else:
        if field in cluster_info.keys:
            kmeans = cluster_info[field]
        else:
            print('error, not such field in clustering')
    data[field + 'Cluster'] = kmeans.predict(data[[field]])
    data = order_cluster(field + 'Cluster', field,data, order)

    if mode == 'auto':
        return data, num_cluster
    else:
        return data


def get_freq(tx_user,transaction):
    tx_frequency = transaction.groupby('customer_id').td.count().reset_index()

    tx_frequency.columns = ['customer_id','Frequency']
    tx_frequency.head()
    tx_user = pd.merge(tx_user, tx_frequency, on='customer_id')

    tx_user = get_clust(tx_user, 'Frequency', params['frequency_cluster'], True)
    return tx_user

def get_profit(tx_user,transaction):
    transaction['standard_cost'] = transaction['standard_cost'].apply(clean_currency).astype('float')
    transaction['profit'] = transaction['list_price'] - transaction['standard_cost']
    tx_profit = transaction.groupby('customer_id').profit.sum().reset_index()
    tx_user = pd.merge(tx_user, tx_profit, on='customer_id')

    return get_clust(tx_user, 'profit', params['profit_cluster'], True)

def get_recency(tx_user,transaction):
    tx_max_purchase = transaction.groupby('customer_id').td.max().reset_index()
    tx_max_purchase.columns = ['customer_id','MaxPurchaseDate']
    tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
    tx_user = pd.merge(tx_user, tx_max_purchase[['customer_id','Recency']], on='customer_id')

    return get_clust(tx_user, 'Recency', params['recency_cluster'], False)

def seg_cluster(tx_user, transaction, back_in_time):
    tx_user['OverallScore'] = tx_user['RecencyCluster']*params['recency_weight'] + tx_user['FrequencyCluster']*params['frequency_weight'] + tx_user['profitCluster']*params['profit_weight']
 
    tx_user = get_clust(tx_user, 'OverallScore', params['seg_cluster'], True)
    tx_user.rename(columns = {'OverallScoreCluster': 'seg_cluster'},
                inplace=True)
    return tx_user

def seg_cluster1(tx_user, transaction, back_in_time):
    decay = 1.084
    dates_customer = transaction['td']
    if back_in_time:
        dates = (dates_customer.max() - dates_customer).dt.days
    else:
        dates = (dates_customer - dates_customer.min()).dt.days
    tx_profit = transaction[['customer_id','profit']]
    tx_profit['cur_value'] = tx_profit['profit'].div(np.power(decay ,(dates.div(10.))))
    tx_profit.dropna(inplace=True)
    tx_user['OverallScore1'] = tx_profit.groupby('customer_id')['cur_value'].sum()
    mean = tx_user['OverallScore1'].mean()
    std = tx_user['OverallScore1'].std()
    tx_user = tx_user[tx_user['OverallScore1'] < mean + 2*std]
 
    tx_user = get_clust(tx_user, 'OverallScore1', 20, True)
    tx_user.rename(columns = {'OverallScore1Cluster': 'seg_cluster'},
                inplace=True)
    return tx_user
            
def get_RFM(transaction, back_in_time):
    user = pd.DataFrame(transaction['customer_id'].unique())
    user.columns = ['customer_id']
    transaction.td = transaction.td.apply(pd.to_datetime)
    user = get_freq(user, transaction)
    user = get_recency(user, transaction)
    user = get_profit(user, transaction)
    # user = seg_cluster(user, transaction, back_in_time)
    user = seg_cluster(user, transaction, back_in_time)
    return user
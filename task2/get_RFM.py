import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import helper
from importlib import reload
reload(helper)
from helper import *
import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from datetime import datetime, timedelta,date
from sklearn.metrics import classification_report,confusion_matrix
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import yaml

with open('param.yaml', 'rb') as f:
    params = yaml.load(f)

def get_clust(data, field, num_cluster, order):
    kmeans = KMeans(n_clusters=num_cluster)
    kmeans.fit(data[[field]])
    data[field + 'Cluster'] = kmeans.predict(data[[field]])
    data.groupby(field + 'Cluster')[field].describe()
    data = order_cluster(field + 'Cluster', field,data, order)
    return data


def get_freq(tx_user,transaction):
    tx_frequency = transaction.groupby('customer_id').transaction_date.count().reset_index()

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
    tx_max_purchase = transaction.groupby('customer_id').transaction_date.max().reset_index()
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
    dates_customer = transaction['transaction_date']
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
    user = get_freq(user, transaction)
    user = get_recency(user, transaction)
    user = get_profit(user, transaction)
    # user = seg_cluster(user, transaction, back_in_time)
    user = seg_cluster1(user, transaction, back_in_time)
    return user
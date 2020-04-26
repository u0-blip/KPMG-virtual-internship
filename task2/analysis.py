# In[1]: import libraries
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import helper
from importlib import reload
import helper
import get_RFM
import plot_coor
reload(helper)
reload(get_RFM)
reload(plot_coor)
from get_RFM import get_RFM, get_clust
from plot_coor import plot_coor


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


# In[2]: Import data

customer = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name="CustomerDemographic", header=1)
new_customer = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name="NewCustomerList", header=1)
transaction = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name="Transactions", header=1)
address = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name="CustomerAddress", header=1)



# In[3]: calculating the RFM values

tx_user = get_RFM(transaction, False)



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

plot_coor(tx_graph, 'Frequency', 'profit','seg_cluster')
plot_coor(tx_graph, 'Recency', 'Frequency','seg_cluster')
plot_coor(tx_graph, 'profit', 'Recency','seg_cluster')

# In[6] 

tx_3m = transaction[(transaction.transaction_date >= date(2017,3,1)) & (transaction.transaction_date < date(2017,6,1))].reset_index(drop=True)
tx_user_3m = get_RFM(tx_3m, True)

tx_6m = transaction[(transaction.transaction_date >= date(2017,6,1)) & (transaction.transaction_date < date(2017,12,1))].reset_index(drop=True)
tx_user_6m = get_RFM(tx_6m, False)

tx_user_6m.rename(columns={'profit':'6 mth profit'}, inplace=True)

tx_user_6m.head()

plot_data = [
    go.Histogram(
        x=tx_user_6m['6 mth profit']
    )
]

plot_layout = go.Layout(
        title='6m profit'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

tx_user_6m = tx_user_6m[tx_user_6m['customer_id'].isin(tx_user_3m['customer_id'])]
tx_merge = pd.merge(tx_user_3m, tx_user_6m[['customer_id', '6 mth profit']], on='customer_id', how='left')

tx_merge = tx_merge.fillna(0)
tx_merge.groupby('seg_cluster')['6 mth profit'].mean()
tx_graph = tx_merge

# plot_coor(tx_graph, 'OverallScore1', '6 mth profit', 'seg_cluster')


tx_merge = tx_merge[tx_merge['6 mth profit']<tx_merge['6 mth profit'].quantile(0.99)]

kmeans = KMeans(n_clusters=5)
kmeans.fit(tx_merge[['6 mth profit']])
tx_merge['LTVCluster'] = kmeans.predict(tx_merge[['6 mth profit']])
tx_merge = helper.order_cluster('LTVCluster', '6 mth profit',tx_merge,True)
tx_merge.groupby('LTVCluster')['6 mth profit'].describe()
tx_cluster = tx_merge.copy()

tx_class = pd.get_dummies(tx_cluster, columns=['profitCluster', 'RecencyCluster', 'FrequencyCluster', 'seg_cluster'])

corr_matrix = tx_merge.corr()
print('cooralation matrix')
print(corr_matrix['LTVCluster'].sort_values(ascending=False))
X = tx_class.drop(['LTVCluster','6 mth profit', 'customer_id', 'Recency', 'profit', 'Frequency', 'OverallScore'],axis=1)
y = tx_class['LTVCluster']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

# In[10]:

for i in range(1, 15):
    ltv_xgb_model = xgb.XGBClassifier(max_depth=i, learning_rate=0.01,objective= 'multi:softprob',n_jobs=-1).fit(X_train, y_train)

    print('max depth' + str(i))
    print('Accuracy of XGB classifier on training set: {:.2f}'
        .format(ltv_xgb_model.score(X_train, y_train)))
    print('Accuracy of XGB classifier on test set: {:.2f}'
        .format(ltv_xgb_model.score(X_test[X_train.columns], y_test)))

    # print('customer percentage sector: ' + str(tx_class.groupby('LTVCluster').customer_id.count()/tx_class.customer_id.count()))

    # y_pred = ltv_xgb_model.predict(X_test)

    # print(classification_report(y_test, y_pred))


# In[10] getting the importance
from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(10,8))
plot_importance(ltv_xgb_model, ax=ax)



# %%

# %%

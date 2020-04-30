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



# # In[2]: Import data

# customer = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name="CustomerDemographic", header=1)
# new_customer = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name="NewCustomerList", header=1)
# transaction = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name="Transactions", header=1)
# address = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name="CustomerAddress", header=1)



# # In[3]: calculating the RFM values

# tx_user = get_RFM.get_RFM(transaction, False)



# # In[4]: Plotting cluster

# plot_data = [
#     go.Histogram(
#         x=tx_user['Recency']
#     )
# ]

# plot_layout = go.Layout(
#         title='Recency'
#     )
# fig = go.Figure(data=plot_data, layout=plot_layout)
# pyoff.iplot(fig)


# plot_data = [
#     go.Histogram(
#         x=tx_user['Frequency']
#     )
# ]

# plot_layout = go.Layout(
#         title='Frequency'
#     )
# fig = go.Figure(data=plot_data, layout=plot_layout)
# pyoff.iplot(fig)


# plot_data = [
#     go.Histogram(
#         x=tx_user['profit']
#     )
# ]

# plot_layout = go.Layout(
#         title='Monetary Value'
#     )
# fig = go.Figure(data=plot_data, layout=plot_layout)
# pyoff.iplot(fig)



# # In[5]: plotting segmentation
# tx_graph = tx_user

# plot_coor.plot_coor(tx_graph, 'Frequency', 'profit','seg_cluster')
# plot_coor.plot_coor(tx_graph, 'Recency', 'Frequency','seg_cluster')
# plot_coor.plot_coor(tx_graph, 'profit', 'Recency','seg_cluster')

## In[6] regression model

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

tx_6m = transaction[(transaction.transaction_date >= pd.to_datetime('2017-6-1')) & (transaction.transaction_date < pd.to_datetime('2017-12-1'))].reset_index(drop=True)
tx_user_6m = get_RFM.get_RFM(tx_6m, back_in_time = False)

tx_user_6m.rename(columns={'profit':'6 mth profit'}, inplace=True)

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

tx_user_6m = tx_user_6m[tx_user_6m['customer_id'].isin(tx_user_3m['customer_id'])]
tx_merge = pd.merge(tx_user_3m, tx_user_6m[['customer_id', '6 mth profit']], on='customer_id', how='left')

tx_merge = tx_merge.fillna(0)
tx_merge.groupby('seg_cluster')['6 mth profit'].mean()
tx_graph = tx_merge

# plot_coor.plot_coor(tx_graph, 'OverallScore1', '6 mth profit', 'seg_cluster')


tx_merge = tx_merge[tx_merge['6 mth profit']<tx_merge['6 mth profit'].quantile(0.99)]


acc = []

kmeans = KMeans(n_clusters=6)
kmeans.fit(tx_merge[['6 mth profit']])
tx_merge['LTVCluster'] = kmeans.predict(tx_merge[['6 mth profit']])
tx_merge = helper.order_cluster('LTVCluster', '6 mth profit',tx_merge,True)
tx_cluster = tx_merge.copy()

y = tx_cluster[['customer_id', 'LTVCluster', 'OverallScore1']]

dummies = ['profitCluster', 'RecencyCluster', 'FrequencyCluster', 'seg_cluster']
tx_class = get_dummies(tx_cluster, dummies)


if not train:
    customer_clean = helper.clean_gender(new_customer)
else:
    customer_clean = helper.clean_gender(customer)

def prep_input(customer_clean):
    customer_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    customer_clean.dropna(how='any', inplace=True)
    customer_clean = helper.get_age(customer_clean)

    customer_clean, num = get_RFM.get_clust(customer_clean, 'tenure', 10, min_cluster = 2, mode='auto')
    print('optimal cluster: ' + str(num))
    customer_clean, num = get_RFM.get_clust(customer_clean, 'Age', 15, min_cluster = 7, mode='auto')
    print('optimal cluster: ' + str(num))
    customer_clean, num = get_RFM.get_clust(customer_clean, 'past_3_years_bike_related_purchases', 20, min_cluster = 5, mode='auto')
    print('optimal cluster: ' + str(num))

    dummies = ['AgeCluster', 'gender', 'wealth_segment', 'owns_car', 'tenureCluster', 'deceased_indicator']
    customer_clean = get_dummies(customer_clean, dummies)
    return customer_clean

customer_clean = prep_input(customer_clean)

if train:
    X_demo = customer_clean[customer_clean['customer_id'].isin(y['customer_id'])]

    # corr_matrix = pd.merge(customer_clean, y, on='customer_id').corr()
    # print('cooralation matrix')
    # print(corr_matrix['LTVCluster'].sort_values(ascending=False).to_string())
    y = y[y['customer_id'].isin(X_demo['customer_id'])]

    model = 'classifier'


    # X_demo = pd.merge(X_demo, tx_class[tx_class['customer_id'].isin(X_demo['customer_id'])], on='customer_id')
    # y_col = y.columns

    # X_demo = pd.merge(X_demo, y, on='customer_id')
    # X_demo = X_demo.sample(frac=1).reset_index(drop=True)
    # y = X_demo[y_col]
    ids = X_demo['customer_id']

    # X_demo.drop(columns=y_col, inplace}=True)


    if model != 'regressor':
        y = y['LTVCluster']
    elif model == 'regressor':
        y = y['OverallScore1']

    X_train, X_test, y_train, y_test = train_test_split(X_demo, y, test_size=0.2, random_state=56)

else:
    X_demo = customer_clean

# In[10]:
if train:
    prev = time()
    if model != 'regressor':
        ltv_xgb_model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1,objective= 'multi:softprob',n_jobs=-1, tree_method='gpu_hist', gpu_id=0).fit(X_train, y_train)
    elif model == 'regressor':
        ltv_xgb_model = xgb.XGBRegressor(max_depth=5, n_estimators=800, learning_rate=0.1,objective='reg:squarederror',n_jobs=-1, tree_method='gpu_hist', gpu_id=0).fit(X_train, y_train)
    after = time()
    print('elapsed: ' + str(after - prev))


print('Accuracy of XGB classifier on training set: {:.2f}'
    .format(ltv_xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
    .format(ltv_xgb_model.score(X_test[X_train.columns], y_test)))

acc.append([ltv_xgb_model.score(X_train, y_train), ltv_xgb_model.score(X_test[X_train.columns], y_test)])

# print('customer percentage sector: ' + str(tx_class.groupby('LTVCluster').customer_id.count()/tx_class.customer_id.count()))



# print(classification_report(y_test, y_pred))



# In[10] getting the importance
from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(10,8))
plot_importance(ltv_xgb_model, ax=ax)

# %% write recommandations to excel
y_pred = ltv_xgb_model.predict(X_demo)
num_customer_rec = 1000
highest_value = np.argsort(-y_pred)
recommand_customer = ids.iloc[highest_value[:num_customer_rec]]
recommand_customer = customer[customer['customer_id'].isin(recommand_customer)]
with pd.ExcelWriter('KPMG_VI_New_raw_data_update_final.xlsx', mode='a') as f:
    recommand_customer.to_excel(f, 'recommandation')

# %%

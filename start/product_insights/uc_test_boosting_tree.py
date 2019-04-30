import pandas as pd
from pandas import ExcelFile
from pandas import ExcelWriter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import export_graphviz
from graphviz import Source
import pydot

df_x_retention = pd.read_excel('/Users/haimo.liu/Desktop/history/etl/yindu_retention_app.xlsx',
                         sheet_name='Sheet2')

df_x_ctr = pd.read_excel('/Users/haimo.liu/Desktop/history/etl/yindu_ctr_app.xlsx',
                         sheet_name='Sheet1')

# print(df_x_ctr.columns)
# print(df_x_retention.columns)

df_merged = df_x_retention.join(df_x_ctr.set_index('date'), on='day_1')

# print(df_merged.columns)

labels = np.array(df_merged['day_2_retention'])
df_select = df_merged.drop(['day_2_retention', 'day_1', 'day_2', 'day_1_UV', 'day_2_UV'], axis=1)
feature_list = list(df_select.columns)

features = np.array(df_select)

labels = np.delete(labels, 0, 0)
features = np.delete(features, 0, 0)

# print(features[0])

train_features, test_features, train_labels, test_labels = \
    train_test_split(features, labels, test_size=0.2, random_state=42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# boost = GradientBoostingRegressor(n_estimators=10, max_depth=3, subsample=0.8)
#
# boost.fit(train_features, train_labels)
#
# predictions = boost.predict(test_features)
#
# weights = list(boost.oob_improvement_)
# print(weights)


boost = GradientBoostingRegressor(n_estimators=10, max_depth=3)

boost.fit(train_features, train_labels)

# boost.fit(train_features, np.ones(train_features.shape[0]))

predictions = boost.predict(test_features)



# errors = abs(predictions - test_labels)
# mape = 100 * (errors / test_labels)
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')


tree = boost.estimators_[0, 0]
# print(tree.tree_.value.shape)
export_graphviz(tree, out_file='boost_tree.dot', feature_names=feature_list)
(graph, ) = pydot.graph_from_dot_file('boost_tree.dot')

s = Source.from_file('boost_tree.dot')
s.view()


importances = list(boost.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
for pair in feature_importances:
    print('Variable: {:20} Importance: {}'.format(*pair))


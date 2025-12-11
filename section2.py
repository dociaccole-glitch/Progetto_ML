import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor



df_tr=pd.read_csv('https_training.csv')
df_te=pd.read_csv('https_test.csv')

df_tr=df_tr.dropna()
df_te=df_te.dropna()

df_tr['label'] = pd.Categorical(df_tr['label']).codes
df_te['label'] = pd.Categorical(df_te['label']).codes


y_target='label'

categorial_features=['time', 'c_ip']

columns_to_drop = [y_target] + categorial_features

X_train = df_tr.drop(columns=columns_to_drop, axis=1) 
y_train = df_tr[y_target]
X_test = df_te.drop(columns=columns_to_drop, axis=1)
y_test = df_te[y_target]

scaler=StandardScaler()
X_train_s = scaler.fit_transform(X_train) 
X_test_s = scaler.transform(X_test)



columns= [i for i in range(X_train_s.shape[1])]
df_tmp = pd.DataFrame(X_train_s, columns=columns)
correlation_matrix = df_tmp.corr().abs()


plt.figure()
sns.heatmap(correlation_matrix, cmap='Blues', vmin=0.8, vmax=1, cbar_kws={'label':'Correlation'})
plt.xlabel('Feature')
plt.ylabel('Feature')
plt.show()


#-----------------------------------------------


# extract features having a correlation > 0.8
c = correlation_matrix[correlation_matrix>0.8]
s = c.unstack()
so = s.sort_values(ascending=False).reset_index()

# get strongly correlatead features removing pairs having correlation = 1 because of the diagonal, i.e., correlation between one feature and itself
so = so[(so[0].isnull()==False) & (so["level_0"] != so["level_1"])]

to_be_deleted = []
candidates = list(so["level_0"])

# get the unique set of features to be deleted. Notice that we discard one feature per time considering the case where a feature is strongly correlated with multiple features
subset_so = so
for candidate in candidates:
    if (candidate in list(subset_so["level_0"])): 
        to_be_deleted.append(candidate)
        subset_so = subset_so[(subset_so["level_0"] != candidate) & (subset_so["level_1"] != candidate)]

# to_be_deleted contains the index of columns that you need to remove from both training and test sets
print(len(to_be_deleted), 'features are removed')

# remove the correlated features from bot sets

# Create a mask for the columns to keep
columns_to_keep = np.ones(X_train_s.shape[1], dtype=bool)
columns_to_keep[to_be_deleted] = False

# Use the mask to select only the columns to keep
X_train_s = X_train_s[:, columns_to_keep]
X_test_s = X_test_s[:, columns_to_keep]





# compute the correlation matrix
columns= [i for i in range(X_train_s.shape[1])]
df_tmp = pd.DataFrame(X_train_s, columns=columns)
correlation_matrix = df_tmp.corr().abs()

# display the heatmap
plt.figure()
sns.heatmap(correlation_matrix, cmap='Blues', vmin=0.8, vmax=1, cbar_kws={'label':'Correlation'})
plt.xlabel('Feature')
plt.ylabel('Feature')
plt.show()






knn = KNeighborsClassifier()

log_reg = LogisticRegression()

rand_forest = RandomForestClassifier()

y_test_flat = y_test.ravel()
y_train_flat = y_train.ravel()

knn.fit(X_train_s, y_train_flat)
log_reg.fit(X_train_s, y_train_flat)
rand_forest.fit(X_train_s, y_train_flat)

y_train_pred_knn = knn.predict(X_train_s)
y_train_pred_lr  = log_reg.predict(X_train_s)
y_train_pred_rf  = rand_forest.predict(X_train_s)

y_test_pred_knn = knn.predict(X_test_s)
y_test_pred_lr  = log_reg.predict(X_test_s)
y_test_pred_rf  = rand_forest.predict(X_test_s)




print(classification_report(y_test_flat, y_test_pred_knn))
print(classification_report(y_train_flat, y_train_pred_knn))

print(classification_report(y_test_flat, y_test_pred_lr))
print(classification_report(y_train_flat, y_train_pred_lr))

print(classification_report(y_test_flat, y_test_pred_rf))
print(classification_report(y_train_flat, y_train_pred_rf))

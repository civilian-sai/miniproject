import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('jm1.csv')
#data = data1.sample(frac=0.1, random_state=42)

defect_true_false = data.groupby('defects')['b'].apply(lambda x: x.count())
print('False: ',defect_true_false[0])
print('True: ',defect_true_false[1])
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
trace = go.Histogram(
x = data.defects,
opacity = 0.75,
name = "Defects",
marker = dict(color = 'green'))
hist_data = [trace]
hist_layout = go.Layout(barmode='overlay',
title = 'Defects',
xaxis = dict(title = 'True - False'),
yaxis = dict(title = 'Frequency'),
)
fig = go.Figure(data = hist_data, layout = hist_layout)
iplot(fig)
data.corr()
from sklearn import preprocessing
scale_v = data[['v']]
scale_b = data[['b']]
minmax_scaler = preprocessing.MinMaxScaler()
28
v_scaled = minmax_scaler.fit_transform(scale_v)
b_scaled = minmax_scaler.fit_transform(scale_b)
data['v_ScaledUp'] = pd.DataFrame(v_scaled)
data['b_ScaledUp'] = pd.DataFrame(b_scaled)
data
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection
x=data.iloc[:, :-10].values
y=data.iloc[:,data.columns =='defects']
x
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
data['total_Op']= labelencoder_y.fit_transform(data['total_Op'])
data['uniq_Opnd']= labelencoder_y.fit_transform(data['uniq_Opnd'])
data['uniq_Op']= labelencoder_y.fit_transform(data['uniq_Op'])
data['total_Opnd']= labelencoder_y.fit_transform(data['total_Opnd'])
y
data.info()
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size = 0.2,
random_state =0)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print("Random Forests Algorithm")
print(classification_report(Y_test, y_pred))
29
print(confusion_matrix(Y_test, y_pred))
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(y_pred,Y_test))



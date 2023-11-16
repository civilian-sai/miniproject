import numpy as np
import pandas as pd
import warnings
import pickle
warnings.filterwarnings("ignore")
from sklearn import model_selection
data = pd.read_csv('jm1.csv')
#print(data.shape[1])
x=data.iloc[:, :-15].values

no=x.shape[1]
print(no)
y=data.iloc[:,data.columns =='defects']
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size = 0.2,
random_state =0)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)
pickle.dump(model,open('model1.pkl','wb'))



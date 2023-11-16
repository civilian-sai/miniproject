import numpy as np
import pandas as pd
import warnings
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

warnings.filterwarnings("ignore")

# Load data
data1 = pd.read_csv('jm1.csv')
data = data1.sample(frac=0.02, random_state=42)

# Extract features and target variable
x = data.iloc[:, :-15].values
y = data['defects']  # Select the 'defects' column

# Check the number of features
no = x.shape[0]
print("Number of features:", no)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=0)

# Create and train the classifier
classifier = SVC(kernel="linear")
classifier.fit(X_train, Y_train)

pickle.dump(classifier,open('model2.pkl','wb'))
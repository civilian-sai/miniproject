import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D

warnings.filterwarnings("ignore")

# Load data
data1 = pd.read_csv('jm1.csv')
data = data1.sample(frac=0.2, random_state=42)

# Extract features and target variable
x = data.iloc[:, :-15].values
y = data['defects']  # Select the 'defects' column

# Check the number of features
no = x.shape[0]
print("Number of features:", no)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(x.shape[1],)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=0)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

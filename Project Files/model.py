import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

crop_data=pd.read_csv("Crop_recommendation.csv")

crop_data.rename(columns = {'label':'Crop'}, inplace = True)





# Shuffling data to remove order effects
from sklearn.utils import shuffle

df  = shuffle(crop_data,random_state=5)

# Selection of Feature and Target variables.

x = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['Crop']

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(target)

y = np.reshape(y, (2200, -1))

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state= 0)

print("x_train :",x_train.shape)
print("x_test :",x_test.shape)
print("y_train :",y_train.shape)
print("y_test :",y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier


knn_clf=KNeighborsClassifier()
model = MultiOutputClassifier(knn_clf, n_jobs=-1)
model.fit(x_train, y_train)

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(random_state=1)
model2 = MultiOutputClassifier(forest, n_jobs=-1)
model2.fit(x_train, y_train)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=6)
model3 = MultiOutputClassifier(clf, n_jobs=-1)
model3.fit(x_train, y_train)

import joblib

joblib.dump(model, "model.pkl")
joblib.dump(le, "le.pkl")
joblib.dump(model2, "model2.pkl")
joblib.dump(model3, "model3.pkl")



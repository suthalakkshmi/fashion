
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r'C:\Users\LENO\Desktop\BEPEC\EndToEnd\Fashion\Fashion_Style.csv')

df = data.copy()
del df['size']
del df['body-shape']
del df['Unnamed: 0']
del df['bra size']
del df['height']

label_encoder = LabelEncoder()
df['Dress-Style'] = label_encoder.fit_transform(df['Dress-Style'].astype(str))


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

model_Fashion = RandomForestClassifier(n_estimators=20)

model_Fashion.fit(X_train, y_train)

print(model_Fashion)


# make predictions
expected = y_test
predicted = model_Fashion.predict(X_test)
# summarize the fit of the model
#Correction
metrics.classification_report(expected, predicted)
metrics.confusion_matrix(expected, predicted)

import pickle

pickle.dump(model_Fashion, open("Model_Fashion.pkl", "wb"))

model = pickle.load(open("Model_Fashion.pkl", "rb"))

print(model.predict([[27.0,38.0,34]]))









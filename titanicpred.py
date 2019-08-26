import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

#store modules in a simple variable name
mlpc=MLPClassifier(hidden_layer_sizes=(10,1), random_state=1, solver='lbfgs', alpha=1e-5, max_iter=300)
le=LabelEncoder()
ssr=StandardScaler()
mlr=MLPRegressor(activation='logistic',  alpha=1e-5, solver='lbfgs',  batch_size=7)
lr=LogisticRegression(penalty='l1', dual=False, max_iter=890)
#read datasets in csv format
train_set=pd.read_csv("titanic/train.csv")
test_set=pd.read_csv("titanic/test.csv")
#delete unncessary columns from train_set
print(test_set.columns.values)
train_set.drop(['Embarked'], axis=1, inplace=True)
train_set.drop(['PassengerId'], axis=1, inplace=True)
train_set.drop(['Cabin'], axis=1, inplace=True)
train_set.drop(['Ticket'], axis=1, inplace=True)
train_set.drop(['Sex'], axis=1, inplace=True)
train_set.drop(['Name'], axis=1, inplace=True)

#replace columns with nan values and empty values
train_set['Age'].fillna(0.0, inplace=True)

print(train_set.isnull().sum())
print(train_set.values)
#split datasets into features and target
X=train_set.iloc[:, 1:7]
y=train_set.iloc[:, 0]

print(X, y)

#encode strings


#save label encoder
pickle.dumps(le)

X_train, X_val_and_test, Y_train, Y_val_and_test=train_test_split(X, y, test_size=0.25)
X_val, X_test, Y_val, Y_test=train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)


ssr.fit_transform(X_train)
ssr.transform(X_test)

mlpc.fit(X,y)
mlr.fit(X, y)
model_accuracy=round(mlpc.score(X, y)*100, 2)
model_accuracy2=round(mlr.score(X, y)*100, 2)
print(f"Model Accuracy :{ model_accuracy}")
print(f"Model Accuracy2 :{ model_accuracy2}")

prediction=mlpc.predict(X_test)
prediction2=mlr.predict
print(prediction)
print(prediction2)

print(classification_report(Y_test, prediction))



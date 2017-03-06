import numpy as np
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('samples/titanic.csv', index_col='PassengerId')

data = data.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)
data = data[np.isnan(data.Age) == False]

survived = data.Survived
data = data.drop(['Survived'], axis=1)

label = LabelEncoder()
label.fit(data.Sex.drop_duplicates())
data.Sex = label.transform(data.Sex)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(data, survived)

importances = clf.feature_importances_
print(importances)
print(data)

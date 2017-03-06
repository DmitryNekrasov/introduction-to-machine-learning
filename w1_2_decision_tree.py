import numpy as np
import pandas
from sklearn.preprocessing import LabelEncoder

data = pandas.read_csv('samples/titanic.csv', index_col='PassengerId')
data = data.drop(['Survived', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)
data = data[np.isnan(data.Age) == False]

label = LabelEncoder()
label.fit(data.Sex.drop_duplicates())
data.Sex = label.transform(data.Sex)

print(data)

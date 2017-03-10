import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data_train = pandas.read_csv('samples/perceptron-train.csv')
data_train_class = data_train.Class
data_train_attributes = data_train.drop(['Class'], axis=1)

data_test = pandas.read_csv('samples/perceptron-test.csv')
data_test_class = data_test.Class
data_test_attributes = data_test.drop(['Class'], axis=1)

clf = Perceptron(random_state=241)
clf.fit(data_train_attributes, data_train_class)
prediction = clf.predict(data_test_attributes)

accuracy = accuracy_score(data_test_class, prediction)
print(accuracy)

scaler = StandardScaler()
data_train_attributes_scaler = scaler.fit_transform(data_train_attributes)
data_test_attributes_scaler = scaler.transform(data_test_attributes)

clf.fit(data_train_attributes_scaler, data_train_class)
prediction = clf.predict(data_test_attributes_scaler)

accuracy_norm = accuracy_score(data_test_class, prediction)
print(accuracy_norm)

diff = accuracy_norm - accuracy
print(diff)

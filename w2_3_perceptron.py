import pandas

data_train = pandas.read_csv('samples/perceptron-train.csv')
data_train_class = data_train.Class
data_train_attributes = data_train.drop(['Class'], axis=1)

data_test = pandas.read_csv('samples/perceptron-test.csv')
data_test_class = data_test.Class
data_test_attributes = data_test.drop(['Class'], axis=1)

print(data_test_attributes)

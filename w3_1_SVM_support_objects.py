import pandas
from sklearn.svm import SVC

data = pandas.read_csv('samples/svm-data.csv', header=None)
data_class = data[0]
data_attributes = data.drop(0, axis=1)

svc = SVC(kernel='linear', C=100000, random_state=241)
svc.fit(data_attributes, data_class)

ans = list(map(lambda v: v + 1, svc.support_))
print(' '.join(map(str, ans)))

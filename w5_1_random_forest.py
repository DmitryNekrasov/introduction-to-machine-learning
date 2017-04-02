import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

data = pandas.read_csv('samples/abalone.csv')
data['Sex'] = data['Sex'].map(lambda v: -1 if v == 'F' else (0 if v == 'I' else 1))
x = data.drop(['Rings'], axis=1)
y = data.Rings

cv = KFold(n_splits=5, shuffle=True, random_state=1)

b = []
for i in range(1, 51):
    clf = RandomForestRegressor(n_estimators=i, random_state=1)
    a = np.mean(cross_val_score(estimator=clf, X=x, y=y, cv=cv, scoring='r2'))
    b.append([i, a])
    print(i, a)

ans = list(filter(lambda v: v[1] > 0.52, b))
print('ans =', ans[0][0])

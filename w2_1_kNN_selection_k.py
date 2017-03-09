import numpy as np
import pandas
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale


def get_best_quality(attributes):
    fold = KFold(n_splits=5, shuffle=True, random_state=42)
    qualities = [(np.average(
        cross_val_score(estimator=KNeighborsClassifier(n_neighbors=k), X=attributes, y=data_class, cv=fold)), k)
                 for k in range(1, 51)]
    return max(qualities)


data = pandas.read_csv('samples/wine.csv')
data_class = data.Class
data_attributes = data.drop(['Class'], axis=1)

best_quality, best_k = get_best_quality(data_attributes)
print(best_k, best_quality)

scale_data = scale(data_attributes, axis=0)
best_quality, best_k = get_best_quality(scale_data)
print(best_k, best_quality)

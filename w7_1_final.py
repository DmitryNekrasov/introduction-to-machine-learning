import datetime
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

data = pandas.read_csv('samples/features.csv', index_col='match_id')
data_size = len(data)

# Проверка выборки на наличие пропусков
pass_result = reversed(sorted(map(lambda nc: (1 - nc[0] / data_size, nc[1]),
                                  filter(lambda nc: nc[0] < data_size,
                                         [(data[col_name].count(), col_name) for col_name in data.columns]))))
print('\n'.join(map(lambda nc: nc[1] + ': ' + str(round(nc[0], 3)), pass_result)))

# Удаление признаков, связанных с итогами матча. Разделение данных на признаки и целевую переменную
X = data.drop(
    ['duration', 'tower_status_radiant', 'tower_status_dire',
     'barracks_status_radiant', 'barracks_status_dire', 'radiant_win'], axis=1)
y = data.radiant_win

# Замена пропусков на нули
X = X.fillna(0)

cv = KFold(n_splits=5, shuffle=True, random_state=241)


# Подход 1: градиентный бустинг "в лоб"
def start_gradient_boosting():
    est_nums = [10, 20, 30, 50, 100, 250]
    means = []
    for estimators_number in est_nums:
        print('\nestimators_number =', estimators_number)
        clf = GradientBoostingClassifier(n_estimators=estimators_number, random_state=241)
        start_time = datetime.datetime.now()
        score = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        mean = np.mean(score)
        means.append(mean)
        print('score =', score)
        print('mean =', mean)
        print('time =', datetime.datetime.now() - start_time)

    plt.plot(est_nums, means)
    plt.xlabel('estimators_number')
    plt.ylabel('mean')
    plt.show()

# start_gradient_boosting()


# Подход 2: логистическая регрессия
def start_logistic_regression():
    scaler = StandardScaler()
    scale_x = scaler.fit_transform(X)
    print(scale_x)

start_logistic_regression()

import datetime
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
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


def plot_chart(x, y, x_label, y_label):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# Подход 1: градиентный бустинг "в лоб"
def start_gradient_boosting(x_in):
    est_nums = [10, 20, 30, 50, 100, 250]
    means = []

    for estimators_number in est_nums:
        print('\nestimators_number =', estimators_number)

        clf = GradientBoostingClassifier(n_estimators=estimators_number, random_state=241)

        start_time = datetime.datetime.now()
        score = cross_val_score(clf, x_in, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        finish_time = datetime.datetime.now() - start_time

        mean = np.mean(score)
        means.append(mean)

        print('score =', score)
        print('mean =', mean)
        print('time =', finish_time)

    return est_nums, means

nums, scores = start_gradient_boosting(X)
plot_chart(nums, scores, 'estimators_number', 'mean')


# Подход 2: логистическая регрессия
def start_logistic_regression(x_in):
    powers = np.power(10.0, np.arange(-5, 6))
    means = []

    for c in powers:
        print('\nC =', c)

        clf = LogisticRegression(C=c, random_state=241, n_jobs=-1)

        start_time = datetime.datetime.now()
        score = cross_val_score(clf, x_in, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        finish_time = datetime.datetime.now() - start_time

        mean = np.mean(score)
        means.append(mean)

        print('score =', score)
        print('mean =', mean)
        print('time =', finish_time)

    return powers, means


scaler = StandardScaler()
scale_x = scaler.fit_transform(X)
c_parameters, scores = start_logistic_regression(scale_x)
plot_chart(np.log10(c_parameters), scores, 'log10(C)', 'mean')
max_score, best_c = max(zip(scores, c_parameters))
print('Best C =', best_c)
print('Max Score =', max_score)


# Удаление категориальных признаков
def remove_category_features(x_in):
    del_list = ['{}{}_hero'.format(name, val) for val in range(1, 6) for name in ['r', 'd']]
    del_list.append('lobby_type')
    x = x_in.drop(del_list, axis=1)
    return x


x_without_category = remove_category_features(X)
scale_x = scaler.fit_transform(x_without_category)
c_parameters, scores = start_logistic_regression(scale_x)
plot_chart(np.log10(c_parameters), scores, 'log10(C)', 'mean')
max_score, best_c = max(zip(scores, c_parameters))
print('Best C =', best_c)
print('Max Score =', max_score)

heroes = pandas.read_csv('samples/heroes.csv')
n_heroes = len(heroes)
print('Heroes number =', n_heroes)

# Формирование мешка слов по героям
X_pick = np.zeros((data.shape[0], n_heroes))
for i, match_id in enumerate(data.index):
    for p in range(5):
        X_pick[i, data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
bow_data = pandas.DataFrame(X_pick, index=data.index)

final_x = pandas.concat([x_without_category, bow_data], axis=1)
scale_x = scaler.fit_transform(final_x)
c_parameters, scores = start_logistic_regression(scale_x)
plot_chart(np.log10(c_parameters), scores, 'log10(C)', 'mean')
max_score, best_c = max(zip(scores, c_parameters))
print('Best C =', best_c)
print('Max Score =', max_score)

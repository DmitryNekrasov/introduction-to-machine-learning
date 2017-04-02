import pandas
import matplotlib.pyplot as plt
from math import exp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


def sigmoid(y):
    return 1 / (1 + exp(-y))


def get_log_loss(model, x, y):
    return [log_loss(y_true=y, y_pred=[sigmoid(v) for v in y_pred]) for y_pred in model.staged_decision_function(x)]


def plot_loss(rate, train_loss, test_loss):
    plt.figure()
    plt.plot(train_loss, 'r')
    plt.plot(test_loss, 'g')
    plt.legend(['train', 'test'])
    plt.savefig('charts/chart_lr' + str(rate) + '.png')


data = pandas.read_csv('samples/gbm-data.csv')
y = data['Activity'].values
x = data.drop(['Activity'], axis=1).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=241)

res = {}

for lr in [1, 0.5, 0.3, 0.2, 0.1]:
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, learning_rate=lr)
    clf.fit(x_train, y_train)

    log_loss_train = get_log_loss(clf, x_train, y_train)
    log_loss_test = get_log_loss(clf, x_test, y_test)
    plot_loss(lr, log_loss_train, log_loss_test)

    min_loss_test_value = min(log_loss_test)
    min_loss_test_index = log_loss_test.index(min_loss_test_value)
    res[lr] = [min_loss_test_index, min_loss_test_value]

print(res)

min_loss_index, min_loss_value = res[0.2]
print(min_loss_index, min_loss_value)

rf_clf = RandomForestClassifier(n_estimators=min_loss_index, random_state=241)
rf_clf.fit(x_train, y_train)
rf_test_loss = log_loss(y_true=y_test, y_pred=rf_clf.predict_proba(x_test)[:, 1])
print(rf_test_loss)

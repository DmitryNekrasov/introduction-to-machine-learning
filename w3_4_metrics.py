import pandas
import sklearn.metrics as skm


def print_scores(s):
    print('\n'.join(map(lambda score: score[1] + ": " + str(score[0]), s)))


def get_max_precision(x, y):
    precision, recall, _ = skm.precision_recall_curve(x, y)
    z, _ = max(filter(lambda v: v[1] >= 0.7, zip(precision, recall)))
    return z


data = pandas.read_csv('samples/classification.csv')

data_true = data[data.true == 1]
tp = len(data_true[data_true.pred == 1])
fn = len(data_true[data_true.pred == 0])
data_false = data[data.true == 0]
fp = len(data_false[data_false.pred == 1])
tn = len(data_false[data_false.pred == 0])
print(tp, fp, fn, tn)

print()

accuracy_score = skm.accuracy_score(data.true, data.pred)
precision_score = skm.precision_score(data.true, data.pred)
recall_score = skm.recall_score(data.true, data.pred)
f1_score = skm.f1_score(data.true, data.pred)
print(' '.join(map(str, map(lambda v: round(v, 3), (accuracy_score, precision_score, recall_score, f1_score)))))

print()

scores_data = pandas.read_csv('samples/scores.csv')
t = scores_data.true

scores = [(skm.roc_auc_score(t, scores_data[name]), name) for name in scores_data.columns[1:]]
print_scores(scores)
max_roc_auc, clf_name = max(scores)
print('max:', max_roc_auc, '(' + clf_name + ')')

print()

precisions = [(get_max_precision(t, scores_data[name]), name) for name in scores_data.columns[1:]]
print_scores(precisions)
max_precision, clf_name = max(precisions)
print('max:', max_precision, '(' + clf_name + ')')

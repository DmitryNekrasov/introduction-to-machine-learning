import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data, newsgroups.target)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, newsgroups.target)

best_accuracy, best_c = gs.best_score_, gs.best_params_['C']
print(best_accuracy, best_c)

svc = SVC(kernel='linear', C=best_c, random_state=241)
svc.fit(X, newsgroups.target)
indices = svc.coef_.indices
data = abs(svc.coef_.data)
a = list(zip(data, indices))
a.sort(reverse=True)

feature_mapping = vectorizer.get_feature_names()
words = sorted([feature_mapping[index] for _, index in a[:10]])
print(' '.join(words))

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

boston = load_boston()
boston_data = scale(boston.data)
boston_target = boston.target

fold = KFold(n_splits=5, shuffle=True, random_state=42)
qualities = [(np.average(cross_val_score(
    estimator=KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p),
    X=boston_data, y=boston_target, cv=fold, scoring='neg_mean_squared_error')), p) for p in np.linspace(1, 10, 200)]
best_quality, best_p = max(qualities)
print(best_p, best_quality)

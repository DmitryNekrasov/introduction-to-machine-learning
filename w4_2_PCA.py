import numpy as np
import pandas
from sklearn.decomposition import PCA

prices_data = pandas.read_csv('samples/close_prices.csv').drop(['date'], axis=1)

pca = PCA(n_components=10)
pca.fit(prices_data)

s = 0
c = 0
for p in pca.explained_variance_ratio_:
    s += p
    c += 1
    if s >= 0.9:
        break
print(s, c)

transform_prices_data = pandas.DataFrame(pca.transform(prices_data))
first_component = transform_prices_data[0]

dji_data = pandas.read_csv('samples/djia_index.csv')
print(np.corrcoef(first_component, dji_data['DJI'])[0][1])

first_component_weights = pca.components_[0]
_, name = max(zip(first_component_weights, prices_data.columns))
print(name)

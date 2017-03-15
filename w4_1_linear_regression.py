import pandas
import re
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


def get_preprocessed_data(data, vec, dict_vec, train=True):
    data.FullDescription = [re.sub('[^a-zA-Z0-9]', ' ', text.lower()) for text in data.FullDescription]
    x_text = vec.fit_transform(data.FullDescription) if train else vec.transform(data.FullDescription)
    data['LocationNormalized'].fillna('nan', inplace=True)
    data['ContractTime'].fillna('nan', inplace=True)
    x_category = dict_vec.fit_transform(
        data[['LocationNormalized', 'ContractTime']].to_dict('records')) if train else dict_vec.transform(
        data[['LocationNormalized', 'ContractTime']].to_dict('records'))
    x = hstack([x_text, x_category])
    return x


tfidf_vec = TfidfVectorizer(min_df=5)
enc = DictVectorizer()

data_train = pandas.read_csv('samples/salary-train.csv')

x_train = get_preprocessed_data(data_train, tfidf_vec, enc, train=True)
y_train = data_train.SalaryNormalized

clf = Ridge(alpha=1, random_state=241)
clf.fit(x_train, y_train)

data_test = pandas.read_csv('samples/salary-test-mini.csv')
x_test = get_preprocessed_data(data_test, tfidf_vec, enc, train=False)
y_test = clf.predict(x_test)
print(' '.join(map(lambda v: str(round(v, 2)), y_test)))

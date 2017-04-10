import pandas

data = pandas.read_csv('samples/features.csv', index_col='match_id')
data_size = len(data)

# Проверка выборки на наличие пропусков
pass_result = reversed(sorted(map(lambda nc: (1 - nc[0] / data_size, nc[1]),
                                  filter(lambda nc: nc[0] < data_size,
                                         [(data[col_name].count(), col_name) for col_name in data.columns]))))
print('\n'.join(map(lambda nc: nc[1] + ': ' + str(round(nc[0], 3)), pass_result)))

# Удаление признаков, связанных с итогами матча. Разделение данных на признаки и целевую переменную
X = data.drop(
    ['duration', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire'],
    axis=1)
y = data.radiant_win

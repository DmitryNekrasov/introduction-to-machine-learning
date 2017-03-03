import pandas

data = pandas.read_csv('samples/titanic.csv', index_col='PassengerId')

sex_count = data['Sex'].value_counts()
male_count = sex_count['male']
female_count = sex_count['female']
print(male_count, female_count)

passengers_number = data.count()['Name']
print(round(data['Survived'].value_counts()[1] / passengers_number * 100, 2))

print(round(data['Pclass'].value_counts()[1] / passengers_number * 100, 2))

print(data['Age'].mean(), data['Age'].median())

print(data.corr()['SibSp']['Parch'])

females_full_name = data[data.Sex == 'female']['Name']
females_first_name = map(lambda name: [name.split('.')[1].split()[0].replace('(', '').replace(')', '')], females_full_name)
females_name_data = pandas.DataFrame.from_records(data=females_first_name)

print(females_name_data[0].value_counts())

import pandas as pd
from sklearn import preprocessing


# Load the merged train and test set
features = pd.read_csv('data/merged.csv')

# Cabin letter
features['cabin_letter'] = features['Cabin'].apply(lambda x: x[0] if isinstance(x, str) else '?')

# Title
features['title'] = features['Name'].str.extract('\w+,\s(\w+)', expand=False)

# Gender
features['is_male'] = (features['Sex'] == 'male')

# Words in name
features['n_words_in_name'] = features['Name'].apply(lambda x: len(x.split(' ')))

# Impute age
features['Age'].fillna(features['Age'].median(), inplace=True)

# Impute fare
features['Fare'].fillna(features['Fare'].mean(), inplace=True)

# Impute embarked with most frequent modality
features['Embarked'].fillna(features['Embarked'].value_counts().index[0], inplace=True)

# Family size
features['family_size'] = features['SibSp'] + features['Parch'] + 1

# Drop unwanted columns
features.drop(['Cabin', 'Name', 'Ticket', 'Sex'], axis='columns', inplace=True)

# Convert bools to ints
for name, series in features.iteritems():
    if series.dtype == 'bool':
        features[name] = series.astype(int)

# Convert ints to floats
for name, series in features.iteritems():
    if series.dtype == 'int':
        features[name] = series.astype(float)

# Convert objects to ints
for name, series in features.iteritems():
    if series.dtype == 'object':
        features[name] = preprocessing.LabelEncoder().fit_transform(series)

# Rename columns
features.rename(columns={'Age': 'age', 'Embarked': 'embarked', 'Fare': 'fare', 'Parch': 'parch',
                'Pclass': 'p_class', 'Sex': 'sex', 'SibSp': 'sib_sp'}, inplace=True)

# Save features
features.to_csv('data/features.csv', index=False)

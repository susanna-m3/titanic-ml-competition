import pandas as pd
import sklearn as sk
train_path = 'data/train.csv'
train_data = pd.read_csv(train_path)

train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
age_bins = [0, 12, 20, 40, 60, 100]        # child, teen, adult, mid-age, senior
age_labels = [0, 1, 2, 3, 4]
train_data['AgeBin'] = pd.cut(train_data['Age'], bins=age_bins, labels=age_labels)

train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())
fare_bins = [-1, 7.91, 14.454, 31, 512]   # low, mid, high, very high
fare_labels = [0, 1, 2, 3]
train_data['FareBin'] = pd.cut(train_data['Fare'], bins=fare_bins, labels=fare_labels)

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)

train_data['Title'] = train_data['Name'].str.extract(r',\s*([^\.]+)\.')

title_map = {
    'Dr': 'Officer', 'Rev': 'Officer', 'Mlle': 'Miss', 'Mme': 'Mrs', 'Major': 'Officer',
    'Col': 'Officer', 'the Countess': 'Noble', 'Capt': 'Officer', 'Ms': 'Miss', 'Lady': 'Noble',
    'Don': 'Noble', 'Jonkheer': 'Noble'}

train_data['Title'] = train_data['Title'].replace(title_map)
train_data['Title'] = train_data['Title'].map({
    'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Noble': 4, 'Officer': 5}).fillna(6)
title_bins = [-1, 0, 1, 2, 3, 4, 5]
title_labels = [0, 1, 2, 3, 4, 5]
train_data['TitleBin'] = pd.cut(train_data['Title'], bins=title_bins, labels=title_labels)

train_data['Embarked'] = train_data['Embarked'].fillna('S')
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
embkd_bins = [-1, 0, 1, 2]
embkd_labels = [0, 1, 2]
train_data['EmbkdBin'] = pd.cut(train_data['Embarked'], bins=embkd_bins, labels=embkd_labels)


X_train = train_data[['Sex', 'AgeBin', 'Pclass', 'FamilySize', 'IsAlone', 'FareBin', 'TitleBin', 'EmbkdBin']] # feature
y_train = train_data['Survived'] # target 

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# test data processing

test_path = 'data/test.csv'
test_data = pd.read_csv(test_path)


test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
age_bins = [0, 12, 20, 40, 60, 100]        # child, teen, adult, mid-age, senior
age_labels = [0, 1, 2, 3, 4]
test_data['AgeBin'] = pd.cut(test_data['Age'], bins=age_bins, labels=age_labels)

test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
fare_bins = [-1, 7.91, 14.454, 31, 512]   # low, mid, high, very high
fare_labels = [0, 1, 2, 3]
test_data['FareBin'] = pd.cut(test_data['Fare'], bins=fare_bins, labels=fare_labels)

test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)

test_data['Title'] = test_data['Name'].str.extract(r',\s*([^\.]+)\.')

title_map = {
    'Dr': 'Officer', 'Rev': 'Officer', 'Mlle': 'Miss', 'Mme': 'Mrs', 'Major': 'Officer',
    'Col': 'Officer', 'the Countess': 'Noble', 'Capt': 'Officer', 'Ms': 'Miss', 'Lady': 'Noble',
    'Don': 'Noble', 'Jonkheer': 'Noble'}

test_data['Title'] = test_data['Title'].replace(title_map)
test_data['Title'] = test_data['Title'].map({
    'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Noble': 4, 'Officer': 5}).fillna(6)
title_bins = [-1, 0, 1, 2, 3, 4, 5]
title_labels = [0, 1, 2, 3, 4, 5]
test_data['TitleBin'] = pd.cut(test_data['Title'], bins=title_bins, labels=title_labels)

test_data['Embarked'] = test_data['Embarked'].fillna('S')
test_data['Embarked'] = test_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
embkd_bins = [-1, 0, 1, 2]
embkd_labels = [0, 1, 2]
test_data['EmbkdBin'] = pd.cut(test_data['Embarked'], bins=embkd_bins, labels=embkd_labels)

X_test = test_data[['Sex', 'AgeBin', 'Pclass', 'FamilySize', 'IsAlone', 'FareBin', 'TitleBin', 'EmbkdBin']] # feature

test_vals = model.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": test_vals
})
submission.to_csv("submission.csv", index=False)

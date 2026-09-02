# import pandas as pd
# train=pd.read_csv("train.csv")
# # print(train.head())
# # print(train.shape)
# # print(train.columns)
# # print(train.info())
# #train.info()
# #print(train.isnull().sum())
# # print(train[['Age', 'Cabin', 'Embarked']].head(10))
# # print(train['Age'].describe())
# # print(train.groupby('Pclass')['Age'].median())
# # print(train['Age'].isnull().sum())
# train['Age'] = train.groupby('Pclass')['Age'].transform(
#     lambda x: x.fillna(x.median())
# )

# # print(train['Age'].isnull().sum())
# # print(train['Cabin'].head(20))
# train = train.drop('Cabin', axis=1)
# # print(train.columns)
# # print(train['Embarked'].isnull().sum())
# # print(train['Embarked'].value_counts())
# train['Embarked'] = train['Embarked'].fillna('S')
# # print(train['Embarked'].isnull().sum())

# train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
# # print(train['Sex'].head())
# train = pd.get_dummies(train, columns=['Embarked'], dtype=int)
# # print(train.head())
# X = train.drop(['Survived', 'Name', 'Ticket'], axis=1)
# y = train['Survived']
# # print(X.shape)
# # print(y.shape)

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# print(X_train.shape)
# print(X_test.shape)

# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print(y_pred)

# from sklearn.metrics import accuracy_score

# accuracy = accuracy_score(y_test, y_pred)

# print("Accuracy:", accuracy)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)








import pandas as pd
train=pd.read_csv("train.csv")

# print(train.head())
# print(train.shape)
# print(train.columns)
# print(train.info())
#train.info()
#print(train.isnull().sum())
# print(train[['Age', 'Cabin', 'Embarked']].head(10))
# print(train['Age'].describe())
# print(train.groupby('Pclass')['Age'].median())
# print(train['Age'].isnull().sum())

train['Age'] = train.groupby('Pclass')['Age'].transform(
    lambda x: x.fillna(x.median())
)

# print(train['Age'].isnull().sum())
# print(train['Cabin'].head(20))

train = train.drop('Cabin', axis=1)

# print(train.columns)
# print(train['Embarked'].isnull().sum())
# print(train['Embarked'].value_counts())

train['Embarked'] = train['Embarked'].fillna('S')

# print(train['Embarked'].isnull().sum())

train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})

# print(train['Sex'].head())

train = pd.get_dummies(train, columns=['Embarked'], dtype=int)

# print(train.head())

X = train.drop(['Survived', 'Name', 'Ticket'], axis=1)
y = train['Survived']

# print(X.shape)
# print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

test = pd.read_csv("test.csv")

print(test.shape)
print(test.head())
test['Age'] = test.groupby('Pclass')['Age'].transform(
    lambda x: x.fillna(x.median())
)
test = test.drop('Cabin', axis=1)
test['Embarked'] = test['Embarked'].fillna('S')
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

test = pd.get_dummies(test, columns=['Embarked'], dtype=int)

print(test.head())
X_test_final = test.drop(['Name', 'Ticket'], axis=1)
print(X.columns)
print(X_test_final.columns)

X_test_final = X_test_final[X.columns]
X_test_final = scaler.transform(X_test_final)
test_pred = model.predict(X_test_final)

print(test_pred)

# Create submission file

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_pred
})

submission.to_csv("submission.csv", index=False)

print(submission.head())
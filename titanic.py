#!/usr/bin/python3
""" machine learning for titanic survival prediction
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
train_data = pd.read_csv('train.csv')
# print(train_data.isnull().sum())
train_data['Age']=train_data['Age'].fillna(train_data['Age'].mean())
# print(train_data.isnull().sum())

test_data = pd.read_csv('test.csv')
# print(test_data.isnull().sum())
test_data['Age']=test_data['Age'].fillna(test_data['Age'].mean())
test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].mean())
# print(test_data.isnull().sum())
# Encoding the 'Sex' column (male: 1, female: 0)
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = label_encoder.fit_transform(test_data['Sex'])


def logistic_regression():
  """ Logistic Regression for titanic survival prediction
  """
  print('logistic regression')
  # Preprocessing
  # Only using the 'Sex', "Age" ,"Pclass" and "Fare" feature for prediction
  # Selecting features and target
  X = train_data[["Sex", "Age" ,"Pclass","Fare"]]  # Features
  Y = train_data['Survived']  # Target

  # Splitting the data into training and testing sets
  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

  # Logistic Regression Training
  model = LogisticRegression()
  model.fit(x_train, y_train)

  # Calculating the accuracy

  accuracy = accuracy_score(model.predict(x_test), y_test)
  print(f'\tlogistic regression accuracy = {accuracy}')

  #creating submission file
  new_predict = model.predict(test_data[["Sex", "Age" ,"Pclass","Fare"]])
  result = pd.DataFrame({'PassengerId': test_data['PassengerId'],
                          'Survived': new_predict})
  result.to_csv('logistic_regression_result.csv', index=False)


if __name__ == '__main__':
  logistic_regression()

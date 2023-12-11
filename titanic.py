#!/usr/bin/python3
""" Logistic Regression for titanic survival prediction
"""
import pandas as pd
# import logistic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


def logistic_regression():
  """ Logistic Regression for titanic survival prediction
  """
  # Preprocessing
  # Only using the 'Sex' feature for prediction
  # Encoding the 'Sex' column (male: 1, female: 0)
  label_encoder = LabelEncoder()
  print(train_data['Sex'].head())
  train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
  print(train_data['Sex'].head())

  # Selecting features and target
  X = train_data[['Sex']]  # Features
  Y = train_data['Survived']  # Target

  # Splitting the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

  # Logistic Regression
  log_reg = LogisticRegression()
  log_reg.fit(X_train, y_train)

  # Predictions
  new_predict = log_reg.predict(label_encoder.fit_transform(test_data['Sex']))

  # Calculating the accuracy
  # accuracy = accuracy_score(y_test, y_pred)
  # print(accuracy)
  #creating submission file
  
  label_encoder = LabelEncoder()
  test_data['Sex'] = label_encoder.fit_transform(test_data['Sex'])
  result = pd.DataFrame({'PassengerId': test_data['PassengerId'],
                          'Sex': test_data['Sex'],
                          'Survived': new_predict})
  result.to_csv('submission.csv', index=False)


if __name__ == '__main__':
  print('logistic regression')
  # print(train_data.head())
  # print(test_data.head())
  logistic_regression()

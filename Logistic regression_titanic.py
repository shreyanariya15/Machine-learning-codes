#Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

#load the titanic dataset
data = pd.read_csv("C:\Desktop\Machine learning\Programs\titanic.csv")

#Dropping irrelevant columns
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df = data

#Handling missing values
imputer = SimpleImputer(strategy='most_frequent')
df['Age'] = imputer.fit_transform(df[['Age']])
df['Embarked'] = imputer.fit_transform(df[['Embarked']])

#Encoding categorical variables
label_encoder = LabelEncoder()
df ['Sex'] = label_encoder.fit_transform(df['Sex'])
df ['Embarked'] = label_encoder.fit_transform(df['Embarked'])

#Splitting the dataset into features and target variable
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Initializing and training the logistic regression model
log_reg = LogisticRegression()

#Training the model
log_reg.fit(X_train, y_train)

#Making predictions
y_pred = log_reg.predict(X_test)

#Evaluting the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

print("\n Classification Report:")
print(classification_report(y_test, y_pred))


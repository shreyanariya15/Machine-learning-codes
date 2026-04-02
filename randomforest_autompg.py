import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the Auto MPG dataset
auto_mpg = pd.read_csv(r"C:\Users\Shreya Nariya\OneDrive\Desktop\Machine learning\Programs\auto-mpg.csv")
auto1 = auto_mpg.drop('car name', axis=1)
print(auto1. isnull().sum())

#replace ? with null values 
auto1.replace('?', np.nan, inplace=True)
print('\n print how many null values are there in each column')
print(auto1.isnull().sum())

# convert object datatype to float 
auto1['horsepower'] = auto1['horsepower'].astype(float)
print('\n datatyps after conversion of horespower to float')
print(auto1.dtypes)
print('\n print mean of each column')
print(auto1.mean())

print('\n after fill null values with mean of the each column')
auto2 = auto1.fillna(auto1.mean())
X = auto2.drop('mpg', axis=1)
y = auto2['mpg']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the Random Forest regressor
rf.fit(X_train, y_train)

# Predict the target variable for test set 
y_pred = rf.predict(X_test)

#Calculate the mean squared error and R-squared score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared Score: {r2}')

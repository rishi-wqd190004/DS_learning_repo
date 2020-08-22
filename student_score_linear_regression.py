# import the packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# reading the dataset
data = pd.read_csv('/home/richi/python_lectures_self/student_scores.csv')
#print(data.head())

# getting the shape of the data
#print(data.shape)

# get the description
#print(data.describe())
#print(data.info())

# visualizing the data
plt.figure(figsize=(16,8))
plt.scatter(data['Hours'], data['Scores'])
plt.title("Shows graph for hours vs score of a student")
plt.xlabel("Hours given by students")
plt.ylabel("Score recieved by students")
plt.show()

# prepare data for linear regression
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

# print(X)
# print(y)

# split the data into test and train for our model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# create the model
model = LinearRegression()

# fitting the model with training data
model.fit(X_train, y_train)

# printing the intercept values
print("intercept value: ", model.intercept_)

# printing the slope value
print("slope value: ", model.coef_)

# predicting using this model
y_pred = model.predict(X_test)

# creating a dataframe to compare the predicted with the actual values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#print(df)

# change the shpae of the predicted output

# visualizing the output 
plt.figure(figsize=(16,8))
plt.scatter(X, y)
plt.plot(X_test, y_pred, color='r')
plt.title("Shows graph for hours vs score of a student")
plt.xlabel("Hours")
plt.ylabel("Students")
plt.show()


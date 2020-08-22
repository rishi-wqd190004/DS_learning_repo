import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# creating data
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

# print(x)
# print(x.shape)
# print(y)
# print(y.shape)

# my model
model = LinearRegression().fit(x, y)

# getting the score of the model
r_sq = model.score(x, y)
print(r_sq)

# getting the score values
print('intercept:', model.intercept_)

# slope values
print('slope:', model.coef_)

# predicting the score
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

# another way to show prediction
y_pred = model.intercept_ + model.coef_ * x
print(y_pred)

# visualize the inputs with result
plt.figure(figsize=(16,8))
plt.scatter(x, y)
plt.plot(x, y_pred)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
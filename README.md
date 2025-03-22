# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program

2. Import numpy as np

3. Plot the points

4. Initialize the program

5. End the program

## Program:
```
Program to implement the linear regression using gradient descent.

Developed by: Jayamani R
RegisterNumber: 212222100014
```
```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("50_Startups.csv")
data.head()

# Plot data
plt.scatter(data['R&D Spend'], data['Profit'])
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Profit Prediction")

# Feature scaling
data['R&D Spend'] = (data['R&D Spend'] - data['R&D Spend'].mean()) / data['R&D Spend'].std()
data['Profit'] = (data['Profit'] - data['Profit'].mean()) / data['Profit'].std()

# Define computeCost and gradientDescent functions
def computeCost(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    square_err = (h - y) ** 2
    return 1 / (2 * m) * np.sum(square_err)

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = X.transpose().dot(predictions - y)
        descent = alpha * (1 / m) * error
        theta -= descent
        J_history.append(computeCost(X, y, theta))
    return theta, J_history

# Predict profit using the trained model
predicted_profit = X.dot(theta)

# Print the predicted profit values
print("Predicted profit values:")
for i in range(len(predicted_profit)):
    print("Predicted profit for R&D Spend ${}: ${}".format(data['R&D Spend'][i], predicted_profit[i][0]))

# Prepare data for computation
X = np.column_stack((np.ones(len(data)), data['R&D Spend']))
y = data['Profit'].values.reshape(-1, 1)
theta = np.zeros((2, 1))

# Compute initial cost
initial_cost = computeCost(X, y, theta)
print("Initial cost:", initial_cost)

# Perform gradient descent
alpha = 0.01
num_iters = 1500
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

# Print optimized parameters
print("Optimized parameters:", theta)

# Plot cost function
plt.figure()
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\\theta)$")
plt.title("Cost function using Gradient Descent")

# Plot the linear fit
plt.figure()
plt.scatter(data['R&D Spend'], data['Profit'])
plt.plot(data['R&D Spend'], X.dot(theta), color='red')
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Profit Prediction")

plt.show()


```
## Output:
![image](https://github.com/user-attachments/assets/e4ce8872-3da9-4ac4-b5d5-5d02538c79b9)
![image](https://github.com/user-attachments/assets/a8b66f5c-4a0d-4a42-94a4-22346dc9c2a8)
![image](https://github.com/user-attachments/assets/86539069-88ce-4bb5-b816-2328ae4c63c9)
![image](https://github.com/user-attachments/assets/f53fa9cc-09a6-4640-85ea-191b40e42781)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

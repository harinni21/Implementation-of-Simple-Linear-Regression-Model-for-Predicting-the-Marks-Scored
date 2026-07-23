## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
## Developed by: Harinni S
## RegisterNumber: 212224060093
```python
# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#-----------------------------------------------------------------------------
# 1) Load dataset
df = pd.read_csv("C:\\Users\\ksvin\\Downloads\\exp_2_dataset_student_scores.csv")   # CSV should have two columns, e.g. "Hours","Scores"
print("First 5 rows:\n", df.head(), "\n")
print("Last 5 rows:\n", df.tail(), "\n")
#-----------------------------------------------------------------------------
# 2) Prepare input (X) and output (Y)
# Assume CSV columns: Hours (feature) and Scores (target)
X = df.iloc[:, :-1].values   # all rows, all columns except last -> shape (n_samples, 1)
Y = df.iloc[:, -1].values    # all rows, last column -> shape (n_samples,)
print("X (features):", X.flatten())
print("Y (targets):", Y)
#-----------------------------------------------------------------------------
# 3) Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))
#-----------------------------------------------------------------------------
# 4) Create and train the model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)   # fit on training data
#-----------------------------------------------------------------------------
# 5) Predict on the test set
Y_pred = regressor.predict(X_test)
print("\nPredicted values:", np.round(Y_pred, 2))
print("Actual values   :", Y_test)
#-------------------------------------------------------------------------------
# 6) Plot training results
plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color="orange", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
#-------------------------------------------------------------------------------
# 7) Plot testing results (use X_test sorted for a nicer line)
order = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[order]
Y_test_sorted = Y_test[order]
Y_pred_sorted = Y_pred[order]
plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
#----------------------------------------------------------------------------------
# 8) Evaluation metrics
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
#--------------------------------------------------------------------------------------
# 9) Example: predict for new students
new_hours = np.array([[2.5], [8.0]])   # shape must be (n_samples, 1)
pred_new = regressor.predict(new_hours)
print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))
#--------------------------------------------------------------------------------------

```



## Output:

<img width="837" height="551" alt="image" src="https://github.com/user-attachments/assets/91f91972-c795-49c5-a5f0-c2275625c55a" />
<img width="923" height="570" alt="image" src="https://github.com/user-attachments/assets/368cdd8c-37fa-419f-874b-9f0fcc4c1ac4" />
<img width="818" height="542" alt="image" src="https://github.com/user-attachments/assets/bfa29814-135b-47b0-a1f3-9a1e4e26907a" />
<img width="793" height="790" alt="image" src="https://github.com/user-attachments/assets/68a20637-321d-48da-99e9-2c93a6970f37" />
<img width="635" height="746" alt="image" src="https://github.com/user-attachments/assets/15fb0962-2d58-4534-b8b2-ebc44a44776b" />
<img width="805" height="492" alt="image" src="https://github.com/user-attachments/assets/a58e373d-f207-4b81-a308-5bda099f610f" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

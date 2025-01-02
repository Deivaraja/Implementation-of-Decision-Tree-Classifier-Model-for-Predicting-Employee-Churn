# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Problem Definition

Identify independent variables (X) such as satisfaction_level, salary, tenure, and the dependent variable (Y), which indicates churn (1 for churned, 0 for retained).

2.Load Dataset

Load the dataset and inspect for missing values, inconsistent data, or irrelevant features.

3.Preprocessing

Handle missing values.
Encode categorical variables (e.g., salary levels as numeric).
Normalize or standardize features if needed.

4.Split Dataset

Split the dataset into training and testing subsets.

5.Initialize Decision Tree Model

Use DecisionTreeClassifier from the sklearn library. Choose parameters like criterion (e.g., gini or entropy) and max_depth as per the problem complexity.

6.Train the Model

Fit the model on the training data.

7.Predict and Evaluate

Use the trained model to predict the target class for the test dataset.
Evaluate performance using metrics like accuracy, confusion matrix, precision, recall, F1-score, and ROC-AUC.

8.Visualization (Optional)

Visualize the decision tree using tools such as Graphviz or matplotlib.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: DEIVARAJA R
RegisterNumber: 24901238
*/
import pandas as pd
 from sklearn.tree import DecisionTreeClassifier,plot_tree
 from sklearn.preprocessing import LabelEncoder
 data=pd.read_csv(r"C:\Users\admin\Downloads\Employee.csv")
 print(data.head())
 print(data.info())
 print(data.isnull().sum())
 data["left"].value_counts()
 
 le=LabelEncoder()
 data["salary"]=le.fit_transform(data["salary"])
#  print(data.head())
 x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
#  print(x.head())    
 y=data["left"]
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
 from sklearn.tree import DecisionTreeClassifier
 dt=DecisionTreeClassifier(criterion="entropy")
 dt.fit(x_train,y_train)
 y_pred=dt.predict(x_test)
 from sklearn import metrics
 accuracy=metrics.accuracy_score(y_test,y_pred)
#  print(accuracy)
 dt.predict([[0.5,0.8,9,260,6,0,1,2]])
 import matplotlib.pyplot as plt
 plt.figure(figsize=(8,6))
 plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
 plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/f4e45bf9-dfcd-4901-9b7d-265917a9f52c)
![image](https://github.com/user-attachments/assets/f509982e-78f0-4501-b5c0-b2a0dcef0c47)
![image](https://github.com/user-attachments/assets/3f90471b-afdb-4433-b077-cb39de2b3099)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

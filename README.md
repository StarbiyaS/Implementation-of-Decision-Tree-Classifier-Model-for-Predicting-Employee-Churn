# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and resd the dataset. 
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the requried values by importing the requried 
   module from sklearn.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Starbiya S
RegisterNumber:212223040208  
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

## Output:
## HEAD() AND INFO():

![Screenshot 2024-09-26 140351](https://github.com/user-attachments/assets/1a3d8a50-10e1-46ad-b395-2b740b82e22a)

## NULL & COUNT:

![Screenshot 2024-09-26 140409](https://github.com/user-attachments/assets/2f3eb7a4-1f1d-46bb-b938-5bacabf5a31d)

![Screenshot 2024-09-26 140430](https://github.com/user-attachments/assets/f8c78eab-d655-4f3e-925f-b08b9610778c)


![Screenshot 2024-09-26 140539](https://github.com/user-attachments/assets/2ca3824c-7150-4bb3-9b08-f6396c69fc42)

## ACCURACY SCORE:

![Screenshot 2024-09-26 140702](https://github.com/user-attachments/assets/b1046091-336e-44b0-94b1-f89dff93aa50)

## DECISION TREE CLASSIFIER MODEL:

![Screenshot 2024-09-26 140722](https://github.com/user-attachments/assets/d03af786-d7dc-4622-bba6-af7d2294f35f)



![Screenshot 2024-09-26 140743](https://github.com/user-attachments/assets/1d8752fb-763a-434c-a906-586699b3d1ec)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

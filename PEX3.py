#Using Scikit Library
from sklearn import datasets
import pandas as pd

#load dataset
data = datasets.load_wine()

#set target 
target = data['target']  #select Decision column   #['class_0', 'class_1', 'class_2']  = [0,1,2]
print('Target == \n' , target)

#inputs data , create dataframe
df = pd.DataFrame(data.data, columns=data.feature_names)
print('Dataframe = \n\n' , df)

inputs = df.values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)


#load model
from sklearn import tree
model = tree.DecisionTreeClassifier()

#fit tha data 
model.fit(X_train, y_train)

#prediction of own data
prediction_data=model.predict(X_test)

# confusion matrix,accuracy,classification_report in sklearn
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

# confusion matrix
matrix = confusion_matrix(y_test,prediction_data)
print('Confusion matrix : \n',matrix)

#Accuracy of model
accuracy = accuracy_score(y_test, prediction_data)
accuracy_percentage = 100 * accuracy
print('Accuracy of model = ' , accuracy_percentage)

#classification_report of model
report=classification_report(y_test, prediction_data) 
print('classification_report = \n', report)
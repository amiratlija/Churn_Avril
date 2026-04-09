import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# Load the data, assuming the file 'customer_churn.csv' is available
data=pd.read_csv('data/customer_churn.csv',sep=',')

# Separate features (x) and target (y)
x = data.drop('Churn', axis=1)
y = data['Churn']

x=data[['Age','Account_Manager', 'Years','Num_Sites']]
y=data['Churn'] 
model=LogisticRegression
#splitter les données en train et test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model=LogisticRegression() 
model.fit(x_train, y_train) 
y_pred=model.predict(x_test) 
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred)) 

joblib.dump(model, 'data/customer_churn_model.pkl')
print('Modele enregistré avec succès')
import pandas as pd #pandas for dataset operations
import numpy as np #numpy for calculations
import matplotlib.pyplot as plt #for drawing graphs

from sklearn.linear_model import LinearRegression #imported linear regression model
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

#path of the cleaned csv
Cleaned_student_performance = "C:/APIIT/My projects/Python projects/Student Performance Predictor/data_set/Cleaned_student_performance.csv"

#error handling incase cleaned dataset is missing
try:
    df = pd.read_csv(Cleaned_student_performance)
except FileNotFoundError:
    print("Please create the cleaned file using data_preprocessor first!")

#features 
x = np.array([df['Hours Studied'],df['Previous Scores'],df['Extracurricular Activities'],df['Sleep Hours'],df['Sample Question Papers Practiced']])
x = x.T #had to transpose because it was (5(r),9000(c)) 
#actual results
y = np.array(df['Performance Index'])

#print(x[:5]) #to check if everything is fine
#print(y[:5])
#print(x.shape)
#print(y.shape)#number of rows is < 10000 because dups were removed

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,
    random_state=42) #splited 20/80 20% for testing

model = LinearRegression() #initialized the model

model.fit(X=x_train,y=y_train) #fitting the training data to the model

#inspect the values of paramaters w,b
print("Coefficient:", model.coef_) #gives the value of the parameter w(weight)
print("Intercept:", model.intercept_)#gives the value of the parameter b(bias)

y_hat = model.predict(X=x_test) #to predcit using the testing data

#Model evaluation
mae = mean_absolute_error(y_true=y_test,y_pred=y_hat)
mse = mean_squared_error(y_true=y_test,y_pred=y_hat)
rmse = np.sqrt(mse)
r2 = r2_score(y_pred=y_hat,y_true=y_test)

print(f"Mean absolute error:{mae:.4f}")
print(f"Mean squared error:{mse:.4f}")
print(f"Root Mean squared error:{rmse:.4f}")
print(f"R2 score:{r2:.4f}")
print(df["Extracurricular Activities"].unique())#to see if the value change worked
print(df.corr(numeric_only=True)["Performance Index"])#check if there is a data leakage
#to check if model underfitting,overfitting

y_train_pred = model.predict(x_train)
train_r2 = r2_score(y_train,y_train_pred)

y_test_pred = model.predict(x_test)
test_r2 = r2_score(y_test,y_test_pred)


print(f"Training R²: {train_r2:.4f}")#if close to one it fits
print(f"Testing R²:  {test_r2:.4f}")

comparision_predictions = pd.DataFrame({
    'Actual_values':y_test[:5],
    'Predicted values':y_hat[:5]
})
#Actual values vs predictions
print(comparision_predictions)

#a manual predicition
#testing manually

x = df.iloc[:1, 0:5] 
y_manual_predicition = np.dot(x,model.coef_) + model.intercept_
print(model.coef_.shape)
print("Manul perdicition:",y_manual_predicition)
print(x)
print(model.predict(x))
print(y_manual_predicition)
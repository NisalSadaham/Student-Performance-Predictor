import pandas as pd #pandas for dataset operations
import numpy as np #numpy for calculations

data_set_path = "C:/APIIT/My projects/Python projects/Student Performance Predictor/data_set/Student_Performance.csv"

#load dataset to a dataframe using pandas
df = pd.read_csv(data_set_path)#loaded the data set into a dataframe
print(df.head())#to check the first 5 rows of the data set

#overview of the dataset
print(df.info)#check data types
print(df.describe())#summary statistics for numerical columns.

#identifying missing values
print("Count null:",df.isnull().sum()) #sum of null values
print("Count is Nan:",df.isna().sum())#sum of Nan values
print("Count duplicates:",df.duplicated().sum())#count number of duplicated values

print(df["Extracurricular Activities"].value_counts()) #checking the values in the extracurricular activites column
#only yes and no was there

#cleaning the dataset

df["Extracurricular Activities"] = np.where(df["Extracurricular Activities"].str.lower() == 'yes',1,0) #replaced yes with 1 and no with 0
print(df["Extracurricular Activities"].head(5))#to confirm
df = df.drop_duplicates()#removing the duplicate values
print("Check for dup after cleaning:",df.duplicated().sum()) #confirming weather there are no duplicated present
cleaned_df = df.copy()

cleaned_df.to_csv("C:/APIIT/My projects/Python projects/Student Performance Predictor/data_set/Cleaned_student_performance.csv",index=False) #to create a cleaned copy of the dataset



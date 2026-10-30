import pandas as pd #pandas for dataset operations
import matplotlib.pyplot as plt #for drawing graphs

data_set_path = "C:/APIIT/My projects/Python projects/ai-engineering-learning/Machine_Learning/linear_regression/Practice_projects/Student Performance Predictor/data_set/Student_Performance.csv"

#load dataset to a dataframe using pandas
df = pd.read_csv(data_set_path)#loaded the data set into a dataframe

print(df["Performance Index"].describe())

#visually represent the performance index using a histogram
plt.hist(df["Performance Index"],bins=20)
plt.xlabel("Performance Index")
plt.ylabel("Number of Students")
plt.title("Distribution of Student Performance")
plt.show()

#to check if there is a relationship between study hours vs performance index
plt.scatter(
    df["Hours Studied"],
    df["Performance Index"],
    marker='.'
)

plt.xlabel("Hours Studied")
plt.ylabel("Performance Index")
plt.title("Hours Studied vs Performance Index")

plt.show()

correlation = df["Hours Studied"].corr(df["Performance Index"]) #checking the correlation 

print(correlation)

#checking the correlation  of other features
features = [
    "Hours Studied",
    "Previous Scores",
    "Sleep Hours",
    "Sample Question Papers Practiced"
]

for feature in features:
    correlation = df[feature].corr(df["Performance Index"])
    print(feature, ":", correlation)

#creating a correlation matrix
correlation_matrix = df.corr(numeric_only=True)

print(correlation_matrix)
#creating a hotmap to show correlation matrix visually
plt.figure(figsize=(10, 7))
plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(correlation_matrix.columns)),correlation_matrix.columns,rotation=45,ha="right")
plt.yticks(range(len(correlation_matrix.columns)),correlation_matrix.columns)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()


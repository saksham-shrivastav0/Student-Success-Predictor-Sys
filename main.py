import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
df = pd.read_csv("Student_Data.csv")
# print(df)

# clean the Data
print(df.info())
print(df.describe())


# Encoding the Data
le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["InternetAccess"] = le.fit_transform(df["InternetAccess"])

df = pd.get_dummies(df, columns=["City"])


# Creating Performance lavel Column

df["Performance_Level"] = ""

df.loc[(df["FinalScore"] > 80) & (df["FinalScore"] <= 100), "Performance_Level"] = (
    "High"
)
df.loc[(df["FinalScore"] >= 60) & (df["FinalScore"] <= 80), "Performance_Level"] = (
    "Medium"
)
df.loc[(df["FinalScore"] < 60), "Performance_Level"] = "Low"
# print(df["Performance_Level"])


# train test split fns
def train_test(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


# plotting
plt.figure(figsize=(10, 6))
plt.ylabel("Attendance / AssignmentsCompleted")
plt.xlabel("FinalScore")
plt.scatter(
    df["Attendance"],
    df["FinalScore"],
    color="blue",
    label="Attendance VS Final Score ",
)

plt.scatter(
    df["AssignmentsCompleted"],
    df["FinalScore"],
    color="red",
    label="Assignments VS Final Score ",
)
plt.tight_layout()
# plt.show()

# scaling features for better performance
X = df[["PreviousScore", "StudyHours", "Attendance", "AssignmentsCompleted"]]
y = df["FinalScore"]
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test(X, y)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# predicting the Final Score using Linear Regression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

Assignments_inp = int(input("How Much Assignment You Completed: "))
Attendance_inp = int(input("Enter the Attendance % of the Student: "))
StudyHours_inp = int(input("Enter the Study Hours of the Student: "))
PrevScore_inp = int(input("Enter the Previous Score of the Student: "))
lr_inp_df = pd.DataFrame(
    [[PrevScore_inp, StudyHours_inp, Attendance_inp, Assignments_inp]],
    columns=["PreviousScore", "StudyHours", "Attendance", "AssignmentsCompleted"],
)
PredictedFinalScore = lr_model.predict(scaler.transform(lr_inp_df))

print(f"Your Result May be : {np.round(PredictedFinalScore[0], 1)}")

# scaling the features for KNN Classifier
X_KNN = df[["PreviousScore", "StudyHours", "Attendance", "AssignmentsCompleted"]]
y_KNN = le.fit_transform(df["Performance_Level"])
scaler = StandardScaler()
X_KNN_train, X_KNN_test, y_KNN_train, y_KNN_test = train_test(X_KNN, y_KNN)

X_KNN_train = scaler.fit_transform(X_KNN_train)
X_KNN_test = scaler.transform(X_KNN_test)

# predicting the Performance Level using KNN Classifier

KN_model = KNeighborsClassifier(n_neighbors=3)
KN_model.fit(X_KNN_train, y_KNN_train)

fs = PredictedFinalScore[0]
i = pd.DataFrame(
    [[PrevScore_inp, StudyHours_inp, Attendance_inp, Assignments_inp]],
    columns=["PreviousScore", "StudyHours", "Attendance", "AssignmentsCompleted"],
)

output = KN_model.predict(scaler.transform(i))[0]
pl_result = "High" if output == 0 else ("Medium" if output == 2 else "Low")

print(f"Your Performance Level May be : {pl_result}")

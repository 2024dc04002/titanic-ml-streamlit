import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

st.title("Titanic Survival Prediction - ML Models")

# Load and clean data
data = pd.read_csv("train.csv")
data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
data["Age"] = data["Age"].fillna(data["Age"].mean())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})

X = data.drop("Survived", axis=1)
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_name = st.selectbox(
    "Select ML Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier()
elif model_name == "KNN":
    model = KNeighborsClassifier(n_neighbors=5)
elif model_name == "Naive Bayes":
    model = GaussianNB()
elif model_name == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model = XGBClassifier(eval_metric="logloss")

model.fit(X_train, y_train)
preds = model.predict(X_test)

st.subheader("Classification Report")
st.text(classification_report(y_test, preds))

st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_test, preds))

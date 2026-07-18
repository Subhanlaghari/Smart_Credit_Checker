import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Dataset ke columns
column_names = [
    "Status_checking_account", "Duration", "Credit_history", "Purpose",
    "Credit_amount", "Savings_account", "Employment_since",
    "Installment_rate", "Personal_status_sex", "Other_debtors",
    "Residence_since", "Property", "Age", "Other_installment_plans",
    "Housing", "Existing_credits", "Job", "Num_dependents",
    "Telephone", "Foreign_worker", "Target"
]

# Dataset load
df = pd.read_csv(
    r"C:\Users\Muhammad Khalid\Desktop\CodeAlpha credit scoring\german (1).data",
    sep=' ',
    header=None,
    names=column_names
)

print(df.head())
print(df.info())
print("Shape:", df.shape)

print("\nTarget Distribution:")
print(df["Target"].value_counts())

# Categorical columns
categorical_cols = df.select_dtypes(include=["object", "string"]).columns
print("\nCategorical Columns:")
print(categorical_cols)

# Har column ka apna alag encoder (yeh important fix hai)
encoders = {}
for col in categorical_cols:
    col_encoder = LabelEncoder()
    df[col] = col_encoder.fit_transform(df[col])
    encoders[col] = col_encoder

print("\nData After Encoding:")
print(df.head())

# Features and Target
X = df.drop("Target", axis=1)
y = df["Target"]

# Column order save karna (web app mein zaroori hoga)
feature_columns = list(X.columns)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining Set:", X_train.shape)
print("Testing Set:", X_test.shape)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(solver="liblinear", max_iter=5000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n------ Model Performance ------")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label=1))
print("Recall   :", recall_score(y_test, y_pred, pos_label=1))
print("F1 Score :", f1_score(y_test, y_pred, pos_label=1))
print("ROC AUC  :", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Sab kuch save karna — model, encoders, scaler, column order
joblib.dump(model, "credit_scoring_model.pkl")
joblib.dump(encoders, "label_encoders.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

print("\n✅ Model, encoders, scaler sab save ho gaye — ab web app ban sakti hai!")
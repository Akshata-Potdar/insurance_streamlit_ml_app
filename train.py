import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
df = pd.read_csv(r"C:\Users\admin\Desktop\PythonProject\PythonProject1\PythonProject\PythonProject\PythonProject\ML_streamlit\INSURANCE (2).csv")

print("Dataset loaded successfully")

print(df.head())
X = df.drop("charges", axis=1)

y = df["charges"]

categorical_columns = [
    "sex",
    "smoker",
    "region"
]

numerical_columns = [
    "age",
    "bmi",
    "children"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

model = LinearRegression()

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

pipeline.fit(X_train, y_train)

print("Model training completed!")

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)
#%%
import pickle

with open("model/insurance_model.pkl", "wb") as file:
    pickle.dump(pipeline, file)

print("Model saved successfully!")
print("File: model/insurance_model.pkl")

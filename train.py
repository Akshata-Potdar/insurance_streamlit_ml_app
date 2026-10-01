import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("Data/Titanic-Dataset.csv")

# 2. Select input features and target
features = [
    "Pclass", "Sex", "Age", "SibSp",
    "Parch", "Fare", "Embarked"
]

X = df[features]
y = df["Survived"]

# 3. Identify numerical and categorical columns
numeric_features = [
    "Pclass", "Age", "SibSp", "Parch", "Fare"
]
categorical_features = ["Sex", "Embarked"]

# 4. Preprocess missing values and categorical data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 5. Create the ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# 6. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. Train the model
model.fit(X_train, y_train)

# 8. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print("Test Accuracy:", round(accuracy, 4))





# 9. Save the complete pipeline
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as model.pkl")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("1. Loading Adult Census Dataset...")
    # Define column names as per UCI documentation
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num", 
        "marital-status", "occupation", "relationship", "race", "sex", 
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    
    # Load data directly from UCI (handling the leading spaces in the CSV)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)
    
    print("2. Cleaning and Preprocessing Data...")
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Separate features (X) and target (y)
    X = df.drop("income", axis=1)
    y = df["income"]
    
    # Encode target variable (<=50K -> 0, >50K -> 1)
    le_y = LabelEncoder()
    y = le_y.fit_transform(y) 
    
    # Encode categorical features
    cat_cols = X.select_dtypes(include=['object']).columns
    le_x = LabelEncoder()
    for col in cat_cols:
        X[col] = le_x.fit_transform(X[col])
        
    print("3. Splitting and Scaling Data...")
    # Split into 80% training and 20% testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("4. Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("5. Evaluating Model...\n")
    y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["<=50K", ">50K"])
    cm = confusion_matrix(y_test, y_pred)
    
    print("-" * 30)
    print(f"Accuracy Score: {acc * 100:.2f}%")
    print("-" * 30)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

if __name__ == '__main__':
    main()

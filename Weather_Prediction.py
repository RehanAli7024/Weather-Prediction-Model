# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
url = "https://bitbucket.org/kayontoga/rattle/raw/master/data/weatherAUS.csv"
df = pd.read_csv(url)

# Data Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['RainTomorrow'])

# Convert categorical columns to numerical
categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features and target
X = df.drop(['Date', 'RainTomorrow'], axis=1)
y = df['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression (Logistic Regression)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

# Support Vector Machine (SVM)
svm = SVC(probability=True)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# Model Evaluation
models = {'Logistic Regression': log_reg_pred, 'KNN': knn_pred, 'Decision Tree': tree_pred, 'SVM': svm_pred}
evaluation_metrics = {'Accuracy': accuracy_score, 'Jaccard Index': jaccard_score, 'F1-Score': f1_score}

for name, predictions in models.items():
    print(f"Evaluation metrics for {name}:")
    for metric_name, metric in evaluation_metrics.items():
        if metric_name == 'Jaccard Index':
            print(f"{metric_name}: {metric(y_test, predictions, average='binary'):.2f}")
        else:
            print(f"{metric_name}: {metric(y_test, predictions):.2f}")

# Log Loss
log_loss_values = {'Logistic Regression': log_loss(y_test, log_reg.predict_proba(X_test)),
                   'KNN': log_loss(y_test, knn.predict_proba(X_test)),
                   'Decision Tree': log_loss(y_test, tree.predict_proba(X_test)),
                   'SVM': log_loss(y_test, svm.predict_proba(X_test))}
for name, value in log_loss_values.items():
    print(f"Log Loss for {name}: {value:.2f}")

# Mean Absolute Error, Mean Squared Error, R2-Score
for name, predictions in models.items():
    print(f"Error metrics for {name}:")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}")
    print(f"R2-Score: {r2_score(y_test, predictions):.2f}")

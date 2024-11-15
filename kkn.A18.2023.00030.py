# knn.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Mempersiapkan Dataset
def load_data():
    iris = load_iris()
    return iris.data, iris.target

# 2. Membagi Dataset menjadi Data Latih dan Data Uji
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# 3. Membuat dan Melatih Model K-NN
def train_knn(X_train, y_train, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

# 4. Memprediksi dan Mengevaluasi Model
def evaluate_model(knn, X_test, y_test):
    y_pred = knn.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return y_pred

# 5. Visualisasi Hasil
def plot_confusion_matrix(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

if __name__ == "__main__":
    # Main execution
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    knn_model = train_knn(X_train, y_train)
    y_pred = evaluate_model(knn_model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)

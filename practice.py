import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

train_data = pd.read_csv("KDDTrain+.txt", header=None)
test_data = pd.read_csv("KDDTest+.txt", header=None)
print(train_data.head())
print(test_data.head())
print(f"Total number of rows in training data: {len(train_data)}")
print(f"Total number of rows in test data: {len(test_data)}")
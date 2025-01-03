# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report, accuracy_score

# train_data = pd.read_csv("KDDTrain+.txt", header=None)
# test_data = pd.read_csv("KDDTest+.txt", header=None)
# print(train_data.head())
# print(test_data.head())

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestClassifier
import joblib
import time

# Load the NSL-KDD dataset (adjust paths as necessary)
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", 
    "logged_in", "num_compromised", "root_shell", "su_attempted", 
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count", 
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", 
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", 
    "dst_host_srv_rerror_rate", "label", "difficulty"
]
print(len(column_names))
#'train.csv' and 'test.csv' with the dataset paths
df = pd.read_csv('KDDTrain+.txt', names=column_names)
df_test = pd.read_csv('KDDTest+.txt', names=column_names)

# Encode categorical features
categorical_columns = ["protocol_type", "service", "flag"]
encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])
    df_test[col] = encoder.transform(df_test[col])

# Encode labels (normal: 0, attack: 1)
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
df_test['label'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Split features and labels
X = df.drop(['label'], axis=1)
y = df['label']

X_test = df_test.drop(['label'], axis=1)
y_test = df_test['label']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
# Define the evaluation function for the genetic algorithm
def evaluate(individual):
    # Extract the selected features
    selected_features = [index for index, value in enumerate(individual) if value == 1]
    if len(selected_features) == 0:
        return 0,
    
    # Train a RandomForestClassifier with the selected features
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X[:, selected_features], y, cv=5, scoring='accuracy')
    return scores.mean(),

# Set up the genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(X[0]))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Run the genetic algorithm
population = toolbox.population(n=50)
ngen = 20
cxpb = 0.5
mutpb = 0.2

algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=True)

# Extract the best individual
best_individual = tools.selBest(population, k=1)[0]
selected_features = [index for index, value in enumerate(best_individual) if value == 1]

# Train the final model with the selected features
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X[:, selected_features], y)

# Evaluate the model on the test set
y_pred = clf.predict(X_test[:, selected_features])
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Export the trained model
model_filename = 'random_forest_model.pkl'
joblib.dump(clf, model_filename)
print(f"Model saved to {model_filename}")
start_time = time.time()

# Run the genetic algorithm
population = toolbox.population(n=50)
ngen = 20
cxpb = 0.5
mutpb = 0.2

algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=True)

# Extract the best individual
best_individual = tools.selBest(population, k=1)[0]
selected_features = [index for index, value in enumerate(best_individual) if value == 1]

# Train the final model with the selected features
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X[:, selected_features], y)

# Evaluate the model on the test set
y_pred = clf.predict(X_test[:, selected_features])
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Export the trained model
model_filename = 'random_forest_model.pkl'
joblib.dump(clf, model_filename)
print(f"Model saved to {model_filename}")

end_time = time.time()
print(f"Time taken to run the model: {end_time - start_time} seconds")
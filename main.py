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

# Genetic Algorithm for Feature Selection
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Create binary representation for each feature
n_features = X.shape[1]
toolbox.register("attr_bool", np.random.randint, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fitness function
def evaluate(individual):
    selected_features = [i for i in range(len(individual)) if individual[i] == 1]
    if len(selected_features) == 0:
        return 0,  # Avoid selecting no features
    
    X_selected = X[:, selected_features]
    classifier = DecisionTreeClassifier(random_state=42)
    scores = cross_val_score(classifier, X_selected, y, cv=5, scoring='accuracy')
    return scores.mean(),

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Genetic Algorithm parameters
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
population = toolbox.population(n=50)
ngen = 20
cxpb = 0.5
mutpb = 0.2

# Run Genetic Algorithm
result = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)

# Extract the best individual
best_individual = tools.selBest(population, k=1)[0]
selected_features = [i for i in range(len(best_individual)) if best_individual[i] == 1]

print(f"Selected Features: {selected_features}")

# Train Decision Tree on selected features
X_train_selected = X[:, selected_features]
X_test_selected = X_test[:, selected_features]

classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train_selected, y)

# Evaluate on test set
y_pred = classifier.predict(X_test_selected)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

# Load the data
url = "https://raw.githubusercontent.com/pratikiiitv/graphicalmodels/main/2020_bn_nb_data.txt"
data = pd.read_csv(url, delimiter="\t")

# Step 2: Learn the structure of the Bayesian Network
hc = HillClimbSearch(data)
best_model = hc.estimate(scoring_method=BicScore(data))

# Step 3: Learn the CPTs for each course node
model = BayesianNetwork(best_model.edges())
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Print the CPTs
print("\nConditional Probability Tables (CPTs):")
for cpd in model.get_cpds():
    print(cpd)

# Step 4: Predict grade in PH100
inference = VariableElimination(model)
query_result = inference.map_query(variables=['PH100'], evidence={'EC100': 'DD', 'IT101': 'CC', 'MA101': 'CD'})
print("\nPredicted grade in PH100:", query_result)

# Step 5: Build and evaluate Naive Bayes classifier (independent features)
X = data.drop(columns=['QP'])
y = data['QP']

# Encode all features and target variable using LabelEncoder
label_encoders = {col: LabelEncoder() for col in X.columns}

X_encoded = X.apply(lambda col: label_encoders[col.name].fit_transform(col))

# Encode the target variable
y_encoded = LabelEncoder().fit_transform(y)

# GaussianNB for independent features
accuracies_gnb = []
for _ in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=None)
    if len(set(y_train)) > 1:  # Ensure there is more than one class in the training set
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_test)
        accuracies_gnb.append(accuracy_score(y_test, y_pred))
print(f"\nAverage accuracy over 20 runs (independent features): {sum(accuracies_gnb) / len(accuracies_gnb)}")

# Step 6: CategoricalNB for dependent features

# Use OneHotEncoder for CategoricalNB to ensure category consistency
# Use OneHotEncoder for CategoricalNB to ensure category consistency
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


# Ensure categories are consistent across training and test sets
accuracies_cnb = []
for _ in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=None)
    
    # Fit OneHotEncoder on the training set and transform both train and test sets
    X_train_encoded = onehot_encoder.fit_transform(X_train)
    X_test_encoded = onehot_encoder.transform(X_test)

    if len(set(y_train)) > 1:  # Ensure there is more than one class in the training set
        nb = CategoricalNB()
        nb.fit(X_train_encoded, y_train)
        y_pred = nb.predict(X_test_encoded)
        accuracies_cnb.append(accuracy_score(y_test, y_pred))

print(f"\nAverage accuracy over 20 runs (dependent features): {sum(accuracies_cnb) / len(accuracies_cnb)}")

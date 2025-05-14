#File: rule_based_system/Decision Tree Model.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Load dataset
df = pd.read_csv(r"./data/cleaned_data.csv")

# Features and target
X = df.drop(columns=['target'])
y = df['target']

# One-hot encode categorical features
X_encoded = pd.get_dummies(X)

# Save feature names for future prediction
joblib.dump(list(X_encoded.columns), 'model_features.joblib')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate model
y_pred = best_model.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print(f"Accuracy:  {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
print(f"Recall:    {recall_score(y_test, y_pred) * 100:.2f}%")
print(f"F1-Score:  {f1_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(best_model, 'heart_disease_decision_tree_model.joblib')
print("Model and features saved successfully.")
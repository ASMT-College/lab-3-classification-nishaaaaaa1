import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Step 0: Create realistic diabetes dataset
# -----------------------------
np.random.seed(42)
data = {
    'Pregnancies': np.random.randint(0, 15, 200),
    'Glucose': np.random.randint(70, 200, 200),
    'BloodPressure': np.random.randint(40, 120, 200),
    'SkinThickness': np.random.randint(10, 60, 200),
    'Insulin': np.random.randint(15, 276, 200),
    'BMI': np.round(np.random.uniform(18, 50, 200), 1),
    'DiabetesPedigreeFunction': np.round(np.random.uniform(0.1, 2.5, 200), 3),
    'Age': np.random.randint(20, 80, 200)
}
df = pd.DataFrame(data)
# Outcome based on glucose and BMI thresholds
df['Outcome'] = ((df['Glucose'] > 125) | (df['BMI'] > 30)).astype(int)

# Save dataset
df.to_csv('diabetes_data.csv', index=False)

df = pd.read_csv('diabetes_data.csv')
X = df.drop(columns='Outcome')
y = df['Outcome']

# Train-test split (30% test size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)


accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Classification Report (Clean):\n")
print(classification_report(y_test, y_pred_nb, zero_division=1))
print(f"Naive Bayes Accuracy: {accuracy_nb:.2f}\n")

nb_predictions_df = X_test.copy()
nb_predictions_df['Actual'] = y_test.values
nb_predictions_df['Predicted_NaiveBayes'] = y_pred_nb
nb_predictions_df.to_csv('naive_bayes_predictions.csv', index=False)
print("CSV file 'naive_bayes_predictions.csv' created successfully!")

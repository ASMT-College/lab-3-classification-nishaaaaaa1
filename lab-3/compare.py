import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

# Load Naive Bayes predictions
nb_df = pd.read_csv('naive_bayes_predictions.csv')
y_test_nb = nb_df['Actual']
y_pred_nb = nb_df['Predicted_NaiveBayes']

# Load Decision Tree predictions
dt_df = pd.read_csv('decision_tree_predictions.csv')
y_test_dt = dt_df['Actual']
y_pred_dt = dt_df['Predicted_DecisionTree']

# Compute Accuracy
accuracy_nb = (y_pred_nb == y_test_nb).mean()
accuracy_dt = (y_pred_dt == y_test_dt).mean()

# Confusion Matrices
cm_nb = confusion_matrix(y_test_nb, y_pred_nb)
cm_dt = confusion_matrix(y_test_dt, y_pred_dt)

# ROC AUC
roc_nb = roc_auc_score(y_test_nb, y_pred_nb)
roc_dt = roc_auc_score(y_test_dt, y_pred_dt)

# Summary
print("\n==================== Comparison: Naive Bayes vs Decision Tree ====================\n")
print(f"Naive Bayes Accuracy       : {accuracy_nb:.2f}")
print(f"Decision Tree Accuracy     : {accuracy_dt:.2f}")
print(f"Naive Bayes ROC AUC        : {roc_nb:.2f}")
print(f"Decision Tree ROC AUC      : {roc_dt:.2f}\n")

print("Confusion Matrix - Naive Bayes:")
print(cm_nb, "\n")

print("Confusion Matrix - Decision Tree:")
print(cm_dt, "\n")
print("Legend: [[TN, FP], [FN, TP]]")
print("\n===============================================================================\n")

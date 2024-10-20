import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

# Load your heart disease dataset (replace 'your_dataset.csv' with your actual dataset file)
df = pd.read_csv("heart_cleveland_upload.csv")
heart_df=df.copy()
# Assume 'target' is the column you want to predict, and other columns are features
heart_df = heart_df.rename(columns={'condition':'target'})
x= heart_df.drop(columns= 'target')
y= heart_df.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.18, random_state=130)




# Train the model with the best hyperparameters on the entire training set
best_rf_model = RandomForestClassifier( criterion='gini',
    n_jobs=-1,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100,
    random_state=123)
best_rf_model.fit(X_train, y_train)
# Predictions on training set
y_train_pred = best_rf_model.predict(X_train)

# Predictions on testing set
y_test_pred = best_rf_model.predict(X_test)

# Calculate training set accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)

# Calculate testing set accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Set Accuracy:", train_accuracy)
print("Testing Set Accuracy:", test_accuracy)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(best_rf_model.get_params())
# Make predictions on the test set
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", accuracy)

# Display classification report for more detailed performance metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import confusion_matrix

# Assuming you have true labels (y_true) and predicted labels (y_pred)
# Example confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Extracting values from the confusion matrix
TN, FP, FN, TP = cm.ravel()

# Calculate sensitivity (recall)
sensitivity = TP / (TP + FN)

# Calculate specificity
specificity = TN / (TN + FP)

print("Sensitivity (Recall):", sensitivity)
print("Specificity:", specificity)

print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))

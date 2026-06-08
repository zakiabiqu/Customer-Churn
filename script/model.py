import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import pickle

# Load dataset
df_load = pd.read_csv('../data/Customer_Churn.csv')

# Visualization of Churn Percentage
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('equal')
labels = ['Yes', 'No']
churn = df_load.Churn.value_counts()
ax.pie(churn, labels=labels, autopct='%.0f%%')
plt.title('Churn Distribution')
plt.show()

# Exploratory Data Analysis (EDA) for Numerical Variables
numerical_features = ['MonthlyCharges', 'TotalCharges', 'tenure']
fig, ax = plt.subplots(1, 3, figsize=(15, 6))
# Plot two overlays of histogram per each numerical_features, using blue and orange colors
df_load[df_load.Churn == 'Yes'][numerical_features].hist(bins=20, color='blue', alpha=0.5, ax=ax)
df_load[df_load.Churn == 'No'][numerical_features].hist(bins=20, color='orange', alpha=0.5, ax=ax)
plt.suptitle('Numerical Features Distribution by Churn')
plt.show()

# Exploratory Data Analysis (EDA) for Categorical Variables
fig, ax = plt.subplots(3, 3, figsize=(14, 12))
sns.countplot(data=df_load, x='gender', hue='Churn', ax=ax[0][0])
sns.countplot(data=df_load, x='Partner', hue='Churn', ax=ax[0][1])
sns.countplot(data=df_load, x='SeniorCitizen', hue='Churn', ax=ax[0][2])
sns.countplot(data=df_load, x='PhoneService', hue='Churn', ax=ax[1][0])
sns.countplot(data=df_load, x='StreamingTV', hue='Churn', ax=ax[1][1])
sns.countplot(data=df_load, x='InternetService', hue='Churn', ax=ax[1][2])
sns.countplot(data=df_load, x='PaperlessBilling', hue='Churn', ax=ax[2][1])
plt.suptitle('Categorical Features Distribution by Churn')
plt.tight_layout()
plt.show()

# Remove unnecessary columns: customerID & UpdatedAt
cleaned_df = df_load.drop(['customerID', 'UpdatedAt'], axis=1)
print("First 5 rows after dropping columns:")
print(cleaned_df.head())

# Convert TotalCharges to numeric
cleaned_df['TotalCharges'] = pd.to_numeric(
    cleaned_df['TotalCharges'],
    errors='coerce'
)
cleaned_df['TotalCharges'] = cleaned_df['TotalCharges'].fillna(0)

# Encode all categorical columns
le = LabelEncoder()

for col in cleaned_df.columns:
    if cleaned_df[col].dtype == 'object' or str(cleaned_df[col].dtype) == 'category':
        cleaned_df[col] = le.fit_transform(cleaned_df[col].astype(str))

print("\nData types after encoding:")
print(cleaned_df.dtypes)

# Splitting Dataset
# Predictor and target
X = cleaned_df.drop('Churn', axis=1)
y = cleaned_df['Churn']
# Splitting train and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Print according to the expected result
print(f"\nNumber of rows and columns of x_train is: {x_train.shape}, while number of rows and columns of y_train is: {y_train.shape}")
print("Percentage of Churn in Training Data:")
print(y_train.value_counts(normalize=True))
print(f"Number of rows and columns of x_test is: {x_test.shape}, while number of rows and columns of y_test is: {y_test.shape}")
print("Percentage of Churn in Testing Data:")
print(y_test.value_counts(normalize=True))

# Training Model: Logistic Regression
log_model = LogisticRegression().fit(x_train, y_train)
print('\nLogistic Regression Model formed is: \n', log_model)

# Predict Train
y_train_pred = log_model.predict(x_train)
print('\nClassification Report Training Model (Logistic Regression):')
print(classification_report(y_train, y_train_pred))

# Confusion matrix as DataFrame
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_train, y_train_pred), 
                                   ('No churn', 'Churn'), 
                                   ('No churn', 'Churn'))
# Plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
plt.title('Confusion Matrix for Training Model\n(Logistic Regression)', fontsize=18, color='darkblue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.tight_layout()
plt.show()

# Predict Test
y_test_pred = log_model.predict(x_test)
print('Classification Report Testing Model (Logistic Regression):')
print(classification_report(y_test, y_test_pred))

# Confusion matrix for test
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_test_pred), 
                                   ('No churn', 'Churn'), 
                                   ('No churn', 'Churn'))
plt.figure()
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
plt.title('Confusion Matrix for Testing Model\n(Logistic Regression)', fontsize=18, color='darkblue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.tight_layout()
plt.show()

# Training Model: Random Forest Classifier
rdf_model = RandomForestClassifier().fit(x_train, y_train)
print('\n', rdf_model)

# Predict Train
y_train_pred = rdf_model.predict(x_train)
print('\nClassification Report Training Model (Random Forest Classifier):')
print(classification_report(y_train, y_train_pred))

# Confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_train, y_train_pred), 
                                   ('No churn', 'Churn'), 
                                   ('No churn', 'Churn'))
plt.figure()
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
plt.title('Confusion Matrix for Training Model\n(Random Forest)', fontsize=18, color='darkblue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.tight_layout()
plt.show()

# Predict Test
y_test_pred = rdf_model.predict(x_test)
print('Classification Report Testing Model (Random Forest Classifier):')
print(classification_report(y_test, y_test_pred))

# Confusion matrix for test
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_test_pred), 
                                   ('No churn', 'Churn'), 
                                   ('No churn', 'Churn'))
plt.figure()
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
plt.title('Confusion Matrix for Testing Model\n(Random Forest)', fontsize=18, color='darkblue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.tight_layout()
plt.show()

# Training Model: Gradient Boosting Classifier
gbt_model = GradientBoostingClassifier().fit(x_train, y_train)
print('\n', gbt_model)

# Predict Train
y_train_pred = gbt_model.predict(x_train)
print('\nClassification Report Training Model (Gradient Boosting):')
print(classification_report(y_train, y_train_pred))

# Confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_train, y_train_pred), 
                                   ('No churn', 'Churn'), 
                                   ('No churn', 'Churn'))
plt.figure()
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
plt.title('Confusion Matrix for Training Model\n(Gradient Boosting)', fontsize=18, color='darkblue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.tight_layout()
plt.show()

# Predict Test
y_test_pred = gbt_model.predict(x_test)
print('Classification Report Testing Model (Gradient Boosting):')
print(classification_report(y_test, y_test_pred))

# Confusion matrix for test
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_test_pred), 
                                   ('No churn', 'Churn'), 
                                   ('No churn', 'Churn'))
plt.figure()
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
plt.title('Confusion Matrix for Testing Model\n(Gradient Boosting)', fontsize=18, color='darkblue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.tight_layout()
plt.show()

# Save Model (logistic regression chosen as best model)
pickle.dump(log_model, open('../output/best_model_churn.pkl', 'wb'))
print("\nModel saved successfully to ../output/best_model_churn.pkl")

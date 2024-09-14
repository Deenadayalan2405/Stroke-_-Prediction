# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sns
import numpy as np
import math
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Load the data
df = pd.read_csv('C:/Users/ssdee/Downloads/stroke_data.csv')

# Checking data info
print(df.info())
print("\n")

# Checking missing values
print(df.isna().sum())
print("\n")

# Checking unique values for each feature
print("Unique values\n", df.nunique())
print("\n")

# Finding outliers in bmi
bmi_outliers = df[df['bmi'] > 50]
print("BMI outliers shape:", bmi_outliers['bmi'].shape)
print("Stroke counts in BMI outliers:\n", bmi_outliers['stroke'].value_counts())

# Replacing outlier entries with mean of bmi
df["bmi"] = df["bmi"].apply(lambda x: df.bmi.mean() if x > 50 else x)

# Replacing null values of bmi with mean of bmi column
df.bmi.replace(to_replace=np.nan, value=df.bmi.mean(), inplace=True)

# Checking data shape and missing values again
print("Data shape after processing:", df.shape)
print("Missing values after processing:\n", df.isna().sum())
print("\n")

# Check the number of male and female gender
print(df.gender.value_counts())
print("\n")

# number of 'other' is very small, converting the value to 'Male'
df['gender'] = df['gender'].replace('Other', 'Male')

# Replacing Male with 1 and Female with 0
df.replace({'gender': {'Male': 1, 'Female': 0}}, inplace=True)

# Replacing No with 0 and Yes with 1
df.replace({'ever_married': {'No': 0, 'Yes': 1}}, inplace=True)

# Replacing Rural with 0 and Urban with 1
df.replace({'Residence_type': {'Rural': 0, 'Urban': 1}}, inplace=True)

# Mapping values of smoking_status to integers
smoking_map = {'Unknown': 0, 'never smoked': 1, 'formerly smoked': 2, 'smokes': 3}
df['smoking_status'] = df['smoking_status'].replace(smoking_map)

# Mapping values of work_type to integers
df['work_type'] = df['work_type'].map({'Private': 0, 'Self-employed': 1, 'children': 2, 'Govt_job': 3,
                                       'Never_worked': 4})

# Checking Datatypes
print(df.dtypes)
print("\n")

# Describing the data
print(df.describe())

# Removing rows where age is less than 20 as it is not relevant for this analysis
df.drop(df[df.age < 20].index, inplace=True)

# Preprocessing
# Drop the ID column as it does not provide any useful information for the model
df = df.drop(['id'], axis=1)

# Remove any remaining rows that contain missing data
df = df.dropna()

# One hot encoding categorical features
df_encoded = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])

# Splitting the data into training and testing sets
X = df.drop(['stroke'], axis=1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fitting the model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Making predictions
y_pred = lr_model.predict(X_test)

# Evaluating the model and generating various plots
# Calculate accuracy, precision, recall and F1 score
print("Logistic Regression Classifier Metrics")
print("\nAccuracy score:", accuracy_score(y_test, y_pred))
print("Precision score:", precision_score(y_test, y_pred, zero_division=1))
print("Recall score:", recall_score(y_test, y_pred, zero_division=1))
print("F1 score:", f1_score(y_test, y_pred, zero_division=1))

# Display the names of the columns
print(df.columns)

# Predict probabilities and calculate ROC curve
y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# calculate AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve and Precision-Recall curve
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

plt.subplot(122)
plt.plot(recall, precision, color='navy', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Precision-Recall Curve')
plt.tight_layout()
plt.show()

# Generate confusion matrix
plt.figure(figsize=(10, 8))
skplt.metrics.plot_confusion_matrix(y_test, y_pred)
plt.show()

# Create the correlation matrix
corr_matrix = df.corr()
# Plot the correlation matrix using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.tight_layout()
plt.show()

# Plot scatter plot of age vs bmi
plt.scatter(df['age'], df['bmi'])
plt.title('Age vs BMI')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.show()

# Plotting various bar charts
plt.figure(figsize=(12, 10))
plt.subplot(421)
sns.countplot(x='gender', hue='stroke', data=df)
plt.title('Stroke Counts by Gender')

plt.subplot(422)
sns.countplot(x='hypertension', hue='stroke', data=df)
plt.title('Stroke Counts by Hypertension')

plt.subplot(423)
sns.countplot(x='heart_disease', hue='stroke', data=df)
plt.title('Stroke Counts by Heart Disease')

plt.subplot(424)
sns.countplot(x='ever_married', hue='stroke', data=df)
plt.title('Stroke Counts by Marital Status')

plt.subplot(425)
sns.countplot(x='work_type', hue='stroke', data=df)
plt.title('Stroke Counts by Work Type')

plt.subplot(426)
sns.countplot(x='Residence_type', hue='stroke', data=df)
plt.title('Stroke Counts by Residence Type')

plt.subplot(427)
sns.countplot(x='smoking_status', hue='stroke', data=df)
plt.title('Stroke Counts by Smoking Status')
plt.tight_layout()

# Plotting histogram of age
plt.figure(figsize=(10, 6))
sns.histplot(x='age', hue='stroke', data=df, kde=True)
plt.title('Stroke Counts by Age')
plt.tight_layout()
plt.show()

# Define the columns to create pie charts for
columns = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status',
           'stroke']

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i, column in enumerate(columns[:4]):
    row = i // 2
    col = i % 2
    axs[row, col].pie(df[column].value_counts(), labels=df[column].value_counts().index, autopct='%1.1f%%')
    axs[row, col].set_title(column)
plt.tight_layout()
plt.show()

fig2, axs2 = plt.subplots(2, 2, figsize=(10, 10))

for i, column in enumerate(columns[4:]):
    row = i // 2
    col = i % 2
    axs2[row, col].pie(df[column].value_counts(), labels=df[column].value_counts().index, autopct='%1.1f%%')
    axs2[row, col].set_title(column)

plt.tight_layout()
plt.show()

# Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print("\nDecision Tree Classifier Metrics")
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Precision score:", precision_score(y_test, y_pred, zero_division=1))
print("Recall score:", recall_score(y_test, y_pred, zero_division=1))
print("F1 score:", f1_score(y_test, y_pred, zero_division=1))

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print("\nRandom Forest Classifier Metrics")
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Precision score:", precision_score(y_test, y_pred, zero_division=1))
print("Recall score:", recall_score(y_test, y_pred, zero_division=1))
print("F1 score:", f1_score(y_test, y_pred, zero_division=1))

# K Nearest Neighbors Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("\nKNN Classifier Metrics")
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Precision score:", precision_score(y_test, y_pred, zero_division=1))
print("Recall score:", recall_score(y_test, y_pred, zero_division=1))
print("F1 score:", f1_score(y_test, y_pred, zero_division=1))

# Support Vector Machine Classifier
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("\nSupport Vector Machine Classifier Metrics")
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Precision score:", precision_score(y_test, y_pred, zero_division=1))
print("Recall score:", recall_score(y_test, y_pred, zero_division=1))
print("F1 score:", f1_score(y_test, y_pred, zero_division=1))

# Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print("\nNaive Bayes Classifier Metrics")
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Precision score:", precision_score(y_test, y_pred, zero_division=1))
print("Recall score:", recall_score(y_test, y_pred, zero_division=1))
print("F1 score:", f1_score(y_test, y_pred, zero_division=1))

# Create a list of classifiers and their names
classifiers = [dtc, rfc, knn, svm, nb]
classifier_names = ['Decision Tree', 'Random Forest', 'KNN', 'Support Vector Machine', 'Naive Bayes']

# Calculate the grid size based on the number of classifiers
num_classifiers = len(classifiers)
grid_size = math.ceil(num_classifiers / 2)

# Create the grid of subplots
fig, axes = plt.subplots(grid_size, 2, figsize=(12, 12))
axes = axes.flatten()

# Iterate over the classifiers and their names
for i, (clf, name) in enumerate(zip(classifiers, classifier_names)):
    # Make predictions using the classifier
    y_pred = clf.predict(X_test)

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create a subplot
    ax = axes[i]
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, ax=ax)
    ax.set_title(name)
    ax.grid(False)

# Remove any extra subplots if the number of classifiers is odd
if num_classifiers % 2 != 0:
    fig.delaxes(axes[-1])

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Cross Validation Scores
print("\nCross Validation Scores")
lr_scores = cross_val_score(lr_model, X, y, cv=5, scoring='accuracy')
dtc_scores = cross_val_score(dtc, X, y, cv=5, scoring='accuracy')
rfc_scores = cross_val_score(rfc, X, y, cv=5, scoring='accuracy')
knn_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
svm_scores = cross_val_score(svm, X, y, cv=5, scoring='accuracy')
nb_scores = cross_val_score(nb, X, y, cv=5, scoring='accuracy')

print("Logistic Regression: ", np.mean(lr_scores))
print("Decision Tree Classifier: ", np.mean(dtc_scores))
print("Random Forest Classifier: ", np.mean(rfc_scores))
print("KNN Classifier: ", np.mean(knn_scores))
print("Support Vector Machine Classifier: ", np.mean(svm_scores))
print("Naive Bayes Classification: ", np.mean(nb_scores), "\n")

# Cross Validation Scores
classifiers = [lr_model, dtc, rfc, knn, svm, nb]
classifier_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN', 'Support Vector Machine',
                    'Naive Bayes']

# Perform cross-validation for each classifier
cv_scores = []
for classifier in classifiers:
    scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
    cv_scores.append(scores)

# Calculate mean accuracy scores
mean_scores = [np.mean(scores) for scores in cv_scores]

# Plotting the cross-validation results
plt.figure(figsize=(10, 6))
bars = plt.bar(classifier_names, mean_scores)
plt.title('Cross Validation Results')
plt.xlabel('Classifier')
plt.ylabel('Accuracy Score')
plt.ylim([0, 1])  # Set the y-axis limits between 0 and 1
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Add labels to each bar
for bar, score in zip(bars, mean_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), '{:.4f}'.format(score),
             ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Get input feature values from the user
gender = int(input("Enter gender (0 for Female, 1 for Male): "))
age = float(input("Enter age: "))
hypertension = int(input("Enter hypertension (0 for No, 1 for Yes): "))
heart_disease = int(input("Enter heart disease (0 for No, 1 for Yes): "))
ever_married = int(input("Enter marital status (0 for No, 1 for Yes): "))
work_type = int(input("Enter work type (0 for Private, 1 for Self-employed, 2 for Children, 3 for Govt_job, "
                      "4 for Never_worked): "))
Residence_type = int(input("Enter residence type (0 for Rural, 1 for Urban): "))
avg_glucose_level = float(input("Enter average glucose level: "))
bmi = float(input("Enter BMI: "))
smoking_status = int(input("Enter smoking status (0 for Unknown, 1 for never smoked, 2 for formerly smoked, "
                           "3 for smokes): "))

# Create a DataFrame with the input values
input_data = pd.DataFrame({'gender': [gender], 'age': [age], 'hypertension': [hypertension],
                           'heart_disease': [heart_disease], 'ever_married': [ever_married],
                           'work_type': [work_type], 'Residence_type': [Residence_type],
                           'avg_glucose_level': [avg_glucose_level], 'bmi': [bmi], 'smoking_status': [smoking_status]
                           })

# Make predictions using the trained logistic regression model
output = lr_model.predict(input_data)

# Print the predicted output
if output[0] == 0:
    print("\nThe predicted output is: No Stroke")
else:
    print("\nThe predicted output is: Stroke")

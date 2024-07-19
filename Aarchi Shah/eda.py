import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn import tree



# Load dataset
# Read the csv file
df = pd.read_csv("crowd_dataset.csv")


'''data = pd.read_csv("crowd_dataset.csv")
df = pd.DataFrame(data)

# Data summary
print(df.head())
print(df.info())
print(df.describe())


# Speed distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Speed'], bins=10, kde=True)
plt.title('Distribution of Speed')
plt.show()

# AgentCount distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['AgentCount'], bins=10, kde=True)
plt.title('Distribution of Agent Count')
plt.show()

# Density distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Density'], bins=10, kde=True)
plt.title('Distribution of Density')
plt.show()

# Scatter plot of X and Y coordinates
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='X', y='Y', hue='label2')
plt.title('Scatter Plot of X and Y Coordinates')
plt.show()

plt.figure(figsize=(15, 10))
plt.subplot(3, 1, 1)
sns.boxplot(data=df, y='Speed')
plt.title('Box Plot of Speed')

plt.subplot(3, 1, 2)
sns.boxplot(data=df, y='AgentCount')
plt.title('Box Plot of Agent Count')

plt.subplot(3, 1, 3)
sns.boxplot(data=df, y='Density')
plt.title('Box Plot of Density')

plt.tight_layout()
plt.show()
'''
# Handling missing values (example)
df.fillna(df.mean(), inplace=True)

# Encode categorical variables if any
df['label2'] = df['label2'].map({'normal': 0, 'abnormal': 1})

# Feature selection
X = df[['X', 'Y', 'Speed', 'Heading', 'AgentCount', 'Density', 'Acc', 'LevelOfCrowdness']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=['normal', 'abnormal'], filled=True)
plt.show()


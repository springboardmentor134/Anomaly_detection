import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("crowd_dataset.csv")

# Treating Null Values
# Drop columns with more than 50% missing values
threshold = len(df) * 0.5
df.dropna(thresh=threshold, axis=1, inplace=True)

# Impute missing numerical values with median
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].median(), inplace=True)


# Handling Outliers using IQR
def cap_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)

for col in ['X', 'Y', 'Speed', 'Acc', 'AgentCount', 'Density']:
    cap_outliers_iqr(df, col)

# Handling Outliers using Z-Score
df[['X', 'Y', 'Speed', 'Acc', 'AgentCount', 'Density']] = df[['X', 'Y', 'Speed', 'Acc', 'AgentCount', 'Density']].apply(zscore)

def flag_outliers_zscore(df, col, threshold=3):
    df[f'{col}_outlier'] = df[col].apply(lambda x: 1 if abs(x) > threshold else 0)

for col in ['X', 'Y', 'Speed', 'Acc', 'AgentCount', 'Density']:
    flag_outliers_zscore(df, col)


    

# Feature Engineering

# Timestamp Conversion
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

# Interaction Features
df['Speed_Density'] = df['Speed'] * df['Density']
df['Speed_AgentCount'] = df['Speed'] * df['AgentCount']

# Binning Continuous Variables
df['Speed_Bin'] = pd.qcut(df['Speed'], q=4, labels=False)
df['Density_Bin'] = pd.qcut(df['Density'], q=4, labels=False)

# Label Encoding
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
df['label2'] = label_encoder.fit_transform(df['label2'])
df['LevelOfCrowdness'] = label_encoder.fit_transform(df['LevelOfCrowdness']) 

# Save the preprocessed DataFrame for future use
df.to_csv("preprocessed_dataset.csv", index=False)

# Define feature columns and target variable
features = ['X', 'Y', 'Speed', 'Heading', 'AgentCount', 'Density', 'Acc', 'LevelOfCrowdness', 'hour', 'day', 'month', 'Speed_Density', 'Speed_AgentCount']
target = 'Severity_level'

# Split the data into features (X) and target (y)
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''# Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# K-Nearest Neighbors Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("K-Nearest Neighbors Accuracy:", accuracy_knn)
print("K-Nearest Neighbors Classification Report:")
print(classification_report(y_test, y_pred_knn))
'''

# Isolation Forest for Anomaly Detection
iso_forest = IsolationForest(contamination=0.1, random_state=42)
df['anomaly_iso'] = iso_forest.fit_predict(X)

# Anomaly Score
df['anomaly_score_iso'] = iso_forest.decision_function(X)

# Summary of results
print("Isolation Forest Anomaly Detection:")
print(df['anomaly_iso'].value_counts())

# Classification Report for Isolation Forest
y_true = df['Severity_level']
y_pred_iso = df['anomaly_iso'].apply(lambda x: 1 if x == -1 else 0)
print("Classification Report for Isolation Forest:")
print(classification_report(y_true, y_pred_iso))

# Local Outlier Factor (LOF) for Anomaly Detection
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
df['anomaly_lof'] = lof.fit_predict(X)

# Anomaly Score
df['anomaly_score_lof'] = lof.negative_outlier_factor_

# Summary of results
print("Local Outlier Factor Anomaly Detection:")
print(df['anomaly_lof'].value_counts())

# Classification Report for Local Outlier Factor
y_pred_lof = df['anomaly_lof'].apply(lambda x: 1 if x == -1 else 0)
print("Classification Report for Local Outlier Factor:")
print(classification_report(y_true, y_pred_lof))


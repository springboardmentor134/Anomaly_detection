import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("crowd_dataset.csv")

# Display the first few rows of the dataset
print(df.head())

# Select the relevant features for correlation
features = ['X', 'Y', 'Speed', 'Heading', 'AgentCount', 'Density', 'Acc', 'LevelOfCrowdness']

# Calculate the correlation matrix
correlation_matrix = df[features].corr()
print(correlation_matrix)

# Create the heatmap with a larger figure size
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Set the title for the heatmap
plt.title('Correlation Heatmap of Features')

# Show the heatmap
plt.show()

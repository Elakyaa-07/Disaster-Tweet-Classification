# Import the pandas library
import pandas as pd

# Read the dataset
data = pd.read_csv("dataset/train.csv")

# Display the first 5 rows
print("\n===== First 5 Rows of the Dataset =====")
print(data.head())

# Display dataset information
print("\n===== Dataset Information =====")
print(data.info())

# Display the column names
print("\n===== Column Names =====")
print(data.columns)

# Display the shape of the dataset
print("\n===== Number of Rows and Columns =====")
print(data.shape)

# Check for missing values
print("\n===== Missing Values =====")
print(data.isnull().sum())

# Display the number of Disaster and Non-Disaster tweets
print("\n===== Disaster vs Non-Disaster Tweets =====")
print(data["target"].value_counts())

# Display basic statistics
print("\n===== Basic Statistics =====")
print(data.describe())

# Display a few tweets with their labels
print("\n===== Sample Tweets =====")
print(data[["text", "target"]].head(10))
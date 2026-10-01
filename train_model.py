import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from utils.preprocessing import clean_text

# Load dataset
data = pd.read_csv("dataset/train.csv")

# Select only required columns
data = data[["text", "target"]]

# Clean tweets
data["clean_text"] = data["text"].apply(clean_text)

# Convert text to numerical features
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data["clean_text"])
y = data["target"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# Save trained model
joblib.dump(model, "models/disaster_model.pkl")

# Save TF-IDF vectorizer
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel Saved Successfully!")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv(r"D:\link academy\Introduction to Machine Learning using Python\ml-product-reviews-project\Data\product_reviews_full.csv")

# drop all rows with missing values
df = df.dropna()

# Convert all sentiment values to lowercase and strip extra spaces
df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()

# Convert column type to 'category'
df['sentiment'] = df['sentiment'].astype('category')

# Drop columns that are not useful for modeling
df = df.drop(columns=['review_uuid', 'product_name', 'product_price'])
 
# Create new column with length of each review_text
df['review_length'] = df['review_text'].astype(str).str.len()

# Define features and label
X = df[["review_title", "review_text", "review_length"]]
y = df["sentiment"]
 
# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "review_title"),
        ("text", TfidfVectorizer(), "review_text"),
        ("length", MinMaxScaler(), ["review_length"])
    ]
)

 
# Define pipeline with the best model (e.g. RandomForestClassifier)
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier())
])

import os

os.makedirs("model", exist_ok=True)

# Train the model on the entire dataset
pipeline.fit(X, y)

# Save the model to a file
os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

model_path = os.path.join(BASE_DIR, "model", "sentiment_model.pkl")
joblib.dump(pipeline, model_path)


print("Model trained and saved as 'model/sentiment_model.pkl'")




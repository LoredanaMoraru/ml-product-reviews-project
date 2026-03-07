import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load CSV
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

# Define pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier())
])

# Build correct model path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "sentiment_model.pkl")

# Train and save
pipeline.fit(X, y)
joblib.dump(pipeline, model_path)

print("Model trained and saved as:", model_path)





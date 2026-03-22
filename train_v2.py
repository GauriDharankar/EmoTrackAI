import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("EmoTrack") \
    .getOrCreate()

# TEXT CLEANING

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# LOAD MENTAL HEALTH DATA

df_mh_spark = spark.read.csv("data/mental_health.csv", header=True, inferSchema=True)

df_mh_spark = df_mh_spark.selectExpr("statement as text", "status as label")

df_mh = df_mh_spark.toPandas()

df_mh = df_mh.rename(columns={
    "statement": "text",
    "status": "label"
})

def map_mh(label):
    label = str(label).lower()
    if "anxiety" in label:
        return "anxiety"
    elif "depression" in label:
        return "depression"
    else:
        return "stress"

df_mh["label"] = df_mh["label"].apply(map_mh)
df_mh = df_mh[["text", "label"]]

# LOAD SENTIMENT140

df_tw_spark = spark.read.csv("data/sentiment140.csv", header=False, inferSchema=True)

df_tw_spark = df_tw_spark.toDF("target", "id", "date", "flag", "user", "text")

df_tw_spark = df_tw_spark.select("text", "target")

df_tw = df_tw_spark.toPandas()

def map_sentiment(label):
    if label == 0:
        return "stress"
    elif label == 4:
        return "normal"
    else:
        return None

df_tw["label"] = df_tw["target"].apply(map_sentiment)
df_tw = df_tw.dropna()
df_tw = df_tw[["text", "label"]]

# LOAD GOEMOTIONS

df_go_spark = spark.read.csv("data/goemotions.csv", header=True, inferSchema=True)

df_go = df_go_spark.toPandas()

def map_goemotion(row):
    # Anxiety
    if row["fear"] == 1 or row["nervousness"] == 1:
        return "anxiety"
    
    # Depression
    elif row["sadness"] == 1 or row["grief"] == 1 or row["remorse"] == 1:
        return "depression"
    
    # Stress
    elif row["anger"] == 1 or row["annoyance"] == 1:
        return "stress"
    
    # Normal
    elif row["joy"] == 1 or row["love"] == 1 or row["optimism"] == 1 or row["gratitude"] == 1:
        return "normal"
    
    else:
        return None

df_go["label"] = df_go.apply(map_goemotion, axis=1)
df_go = df_go.dropna(subset=["label"])
df_go = df_go[["text", "label"]]

# COMBINE ALL DATA

df = pd.concat([df_mh, df_tw, df_go], ignore_index=True)

df = df.dropna()
df = df[df["text"].str.len() > 20]

df = df.sample(n=300000, random_state=42)

from sklearn.utils import resample

df_normal = df[df["label"] == "normal"]
df_stress = df[df["label"] == "stress"]
df_anxiety = df[df["label"] == "anxiety"]
df_depression = df[df["label"] == "depression"]

# Upsample minority classes
df_anxiety_up = resample(df_anxiety, replace=True, n_samples=50000, random_state=42)
df_depression_up = resample(df_depression, replace=True, n_samples=50000, random_state=42)

# Downsample majority classes
df_normal_down = resample(df_normal, replace=False, n_samples=100000, random_state=42)
df_stress_down = resample(df_stress, replace=False, n_samples=100000, random_state=42)

# Combine
df = pd.concat([
    df_normal_down,
    df_stress_down,
    df_anxiety_up,
    df_depression_up
])

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

print("\nFinal Dataset Size:", df.shape)
print("\nLabel Distribution:\n", df["label"].value_counts())

# CLEAN TEXT

df["clean_text"] = df["text"].apply(clean_text)

# TF-IDF

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=12000,
    min_df=3,
    max_df=0.85
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# TRAIN MODEL

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearSVC(class_weight='balanced')
model.fit(X_train, y_train)

# EVALUATION

y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# SAVE MODEL

pickle.dump(model, open("model/model_v2.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer_v2.pkl", "wb"))

print("\nModel v2 saved successfully!")

# TEST

def predict_text(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

print("\nTest Predictions:")
print("I feel nervous:", predict_text("I feel very nervous"))
print("I feel hopeless:", predict_text("I feel hopeless and tired"))
print("I feel good:", predict_text("I feel happy and relaxed"))
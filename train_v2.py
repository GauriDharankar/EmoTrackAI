import pandas as pd
import re
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from explainability import predict_with_explanation

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# TEXT CLEANING

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# LOAD MENTAL HEALTH DATA

df_mh = pd.read_csv(DATA_DIR / "mental_health.csv", usecols=["statement", "status"])
df_mh = df_mh.rename(columns={"statement": "text", "status": "label"})

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

df_tw = pd.read_csv(
    DATA_DIR / "sentiment140.csv",
    header=None,
    names=["target", "id", "date", "flag", "user", "text"],
    usecols=["target", "text"],
    encoding="latin-1",
)

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

df_go = pd.read_csv(DATA_DIR / "goemotions.csv")

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

with (MODEL_DIR / "model_v2.pkl").open("wb") as model_file:
    pickle.dump(model, model_file)
with (MODEL_DIR / "vectorizer_v2.pkl").open("wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nModel v2 saved successfully!")

print("\nTest Predictions:")
sample_texts = [
    "I feel very nervous",
    "I feel hopeless and tired",
    "I feel happy and relaxed",
]

for sample_text in sample_texts:
    explanation = predict_with_explanation(sample_text, model, vectorizer)
    print(
        f"{sample_text}: {explanation['label']} "
        f"({explanation['confidence'] * 100:.1f}% confidence)"
    )
    print("Keywords:", ", ".join(explanation["keywords"]) or "No strong keywords")

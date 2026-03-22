import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# 🔹 STEP 1: Load and combine datasets
files = [
    "data/LD DA 1.csv",
    "data/LD EL1.csv",
    "data/LD PF1.csv",
    "data/LD TS 1.csv"
]

dfs = []

for file in files:
    df = pd.read_csv(file)

    print(f"\nFile: {file}")
    print("Columns:", df.columns)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Use correct columns
    df = df.rename(columns={
        "selftext": "text",
        "label": "label"   # already correct but keep for safety
    })

    # Keep only needed columns
    df = df[["text", "label"]]

    dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

# Remove empty rows
df = df.dropna(subset=["text", "label"])

# convert to lowercase and remove extra spaces
df["label"] = df["label"].str.lower().str.strip()

def clean_label(label):
    if "anxiety" in label:
        return "anxiety"
    elif "depression" in label:
        return "depression"
    elif "stress" in label or "trauma" in label:
        return "stress"
    elif "drug" in label:
        return "stress"
    elif "personality" in label:
        return "normal"
    elif "early life" in label:
        return "normal"
    else:
        return "normal"

df["label"] = df["label"].apply(clean_label)


print("\nClean Labels:\n", df["label"].value_counts())

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

df["clean_text"] = df["text"].apply(clean_text)

def rule_boost(text):
    positive_words = ["fine", "okay", "good", "happy", "relaxed"]

    text = text.lower()
    if any(word in text for word in positive_words):
        return "normal"
    return None

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    ngram_range=(1,3),   
    max_features=8000,
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

from sklearn.utils import resample

df_stress = df[df["label"] == "stress"]
df_normal = df[df["label"] == "normal"]

df_stress_down = resample(df_stress,
                         replace=False,
                         n_samples=len(df_normal),
                         random_state=42)

df = pd.concat([df_stress_down, df_normal])

df = df[df["text"].str.len() > 20]

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearSVC(class_weight='balanced')
model.fit(X_train, y_train)

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("\nModel saved successfully!")

def predict_text(text):
    rule = rule_boost(text)
    if rule:
        return rule

    text = clean_text(text)
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

print("\nTest Prediction:")
print(predict_text("I feel very stressed and tired"))
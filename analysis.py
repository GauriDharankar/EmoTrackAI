import pickle
import re
import matplotlib.pyplot as plt

# LOAD MODEL

model = pickle.load(open("model/model_v2.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer_v2.pkl", "rb"))

# CLEAN TEXT

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# PREDICTION

def predict_text(text):
    text_clean = clean_text(text)
    vec = vectorizer.transform([text_clean])
    pred = model.predict(vec)[0]

    pred = refine_prediction(text, pred)

    return pred

# TIMELINE ANALYSIS

def analyze_timeline(texts):

    results = []
    scores = []

    shift = None

    # Updated 4-class scoring
    score_map = {
        "normal": 1,
        "stress": 2,
        "anxiety": 3,
        "depression": 4
    }

    for t in texts:
        pred = predict_text(t)
        results.append(pred)
        scores.append(score_map[pred])

    # TREND DETECTION

    if all(s == scores[0] for s in scores):
        trend = "Stable"
    elif scores == sorted(scores) and len(set(scores)) > 1:
        trend = "Worsening Mental State"
    elif scores == sorted(scores, reverse=True):
        trend = "Improving"
    else:
        trend = "Fluctuating"

    # SHIFT DETECTION

    if len(scores) >= 2:
        if scores[-2] >= 3 and scores[-1] == 1:
            shift = "Sudden Positive Shift"
        elif scores[-2] == 1 and scores[-1] >= 3:
            shift = "Sudden Negative Shift"

    # RISK SCORING

    avg_score = sum(scores) / len(scores)

    if avg_score >= 3.5:
        risk = "Critical Risk"
    elif avg_score >= 2.5:
        risk = "High Risk"
    elif avg_score >= 1.8:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return results, trend, shift, risk, scores

# Handling edge cases
def rule_refinement(text, pred):
    text = text.lower()

    if "hopeless" in text or "empty" in text or "worthless" in text:
        return "depression"
    
    if "nervous" in text or "panic" in text:
        return "anxiety"
    
    return pred

def refine_prediction(text, pred):
    text = text.lower()

    # Anxiety keywords
    if any(word in text for word in ["nervous", "panic", "anxious", "restless"]):
        return "anxiety"
    
    # Depression keywords
    if any(word in text for word in ["hopeless", "empty", "worthless", "tired of life"]):
        return "depression"

    return pred

# GRAPH FUNCTION

def plot_timeline(scores):

    plt.figure()
    plt.plot(scores, marker='o')

    # Add value labels
    for i, val in enumerate(scores):
        plt.text(i, val + 0.1, str(val), ha='center')

    plt.title("Mental Health Trend Analysis")
    plt.xlabel("Time Sequence")
    plt.ylabel("Mental State Level")

    plt.grid(True)

# EXPLAINABLE AI

def explain_prediction(text):

    text_clean = clean_text(text)
    vec = vectorizer.transform([text_clean])

    feature_names = vectorizer.get_feature_names_out()
    weights = vec.toarray()[0]

    top_indices = weights.argsort()[-5:][::-1]

    keywords = [feature_names[i] for i in top_indices if weights[i] > 0]

    return keywords

# SUGGESTIONS

def get_suggestion(risk):

    if "Critical" in risk:
        return "Immediate support recommended. Please reach out to a trusted person or counselor."
    
    elif "High" in risk:
        return "Consider talking to someone you trust or seeking help."
    
    elif "Medium" in risk:
        return "Take a break and monitor your mental well-being."
    
    else:
        return "You seem to be doing well. Keep maintaining a healthy routine."

# TEST (OPTIONAL)

if __name__ == "__main__":

    texts = [
        "I feel okay today",
        "I am getting anxious",
        "I feel very nervous and overwhelmed",
        "I feel completely hopeless"
    ]

    results, trend, shift, risk, scores = analyze_timeline(texts)

    print("Predictions:", results)
    print("Trend:", trend)
    print("Shift:", shift)
    print("Risk:", risk)

    print("\nKeywords:", explain_prediction(texts[-1]))

    print("\nSuggestion:", get_suggestion(risk))

    plot_timeline(scores)
    plt.show()
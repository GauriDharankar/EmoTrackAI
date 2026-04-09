import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from explainability import predict_with_explanation

# LOAD MODEL

BASE_DIR = Path(__file__).resolve().parent

with (BASE_DIR / "model" / "model_v2.pkl").open("rb") as model_file:
    model = pickle.load(model_file)
with (BASE_DIR / "model" / "vectorizer_v2.pkl").open("rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# PREDICTION

def predict_text(text):
    explanation = predict_with_explanation(text, model, vectorizer)
    return explanation["label"], explanation

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
        pred, explanation = predict_text(t)
        results.append(explanation)
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

    print("Predictions:")
    for text, result in zip(texts, results):
        print(
            f"- {text} -> {result['label']} "
            f"({result['confidence'] * 100:.1f}% confidence)"
        )
        print("  Keywords:", ", ".join(result["keywords"]) or "No strong keywords")

    print("Trend:", trend)
    print("Shift:", shift)
    print("Risk:", risk)

    print("\nSuggestion:", get_suggestion(risk))

    plot_timeline(scores)
    plt.show()

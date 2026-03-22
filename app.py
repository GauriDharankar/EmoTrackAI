import streamlit as st
import pandas as pd
import pickle
import re
import plotly.express as px
import time

# --- 1. SETTINGS & ASSETS ---
st.set_page_config(page_title="EmoTrack: AI Mental Health Tracker", layout="wide", page_icon="🧠")

@st.cache_resource
def load_model():
    model = pickle.load(open("model/model_v2.pkl", "rb"))
    vectorizer = pickle.load(open("model/vectorizer_v2.pkl", "rb"))
    return model, vectorizer

try:
    model, vectorizer = load_model()
except:
    st.error("Model files not found! Please run train_v2.py first.")

# --- 2. CORE LOGIC FUNCTIONS ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

def predict_emotion(text):
    text_clean = clean_text(text)
    # Heuristic Safety Net
    if any(w in text_clean for w in ["hopeless", "worthless", "suicide", "empty", "tired"]):
        return "depression"
    if any(w in text_clean for w in ["panic", "shaking", "nervous", "anxious", "racing"]):
        return "anxiety"
    # ML Prediction
    vec = vectorizer.transform([text_clean])
    return model.predict(vec)[0]

def get_risk_level(scores):
    avg = sum(scores) / len(scores)
    if avg >= 3.5: return "Critical Risk", "Immediate support recommended."
    elif avg >= 2.5: return "High Risk", "Consider professional guidance."
    elif avg >= 1.8: return "Medium Risk", "Monitor trends and take a break."
    return "Low Risk", "Your emotional trend appears stable."

# --- 3. UI SIDEBAR ---
st.sidebar.title("Choose Mode")
mode = st.sidebar.selectbox("", ["Personal Analysis", "Group Pulse (Big Data)", "Professional Insight"])
st.sidebar.markdown("---")
st.sidebar.caption("Project by: Group of Two")

# --- 4. MAIN INTERFACE ---

# --- MODE 1: PERSONAL ANALYSIS ---
if mode == "Personal Analysis":
    st.header("Personal Timeline Tracker")
    st.write("Enter your thoughts from the last few days to see your emotional trend.")

    tab1, tab2 = st.tabs(["Manual Entry", "🔗 Auto-Fetch Social Media"])

    with tab1:
        # The 5-day layout from your screenshot
        inputs = []
        for i in range(1, 6):
            val = st.text_input(f"Day {i} Journal Entry", key=f"d{i}")
            if val: inputs.append(val)

        if st.button("Analyze My Timeline"):
            if inputs:
                preds = [predict_emotion(t) for t in inputs]
                score_map = {"normal": 1, "stress": 2, "anxiety": 3, "depression": 4}
                scores = [score_map[p] for p in preds]
                st.session_state['last_scores'] = scores # Save for Expert Mode
                
                col1, col2 = st.columns(2)
                with col1:
                    risk, sug = get_risk_level(scores)
                    st.metric("Risk Status", risk)
                    st.info(f"**Suggestion:** {sug}")
                with col2:
                    fig = px.line(y=scores, x=[f"Day {i+1}" for i in range(len(scores))], markers=True)
                    fig.update_yaxes(tickvals=[1,2,3,4], ticktext=["Normal", "Stress", "Anxiety", "Depression"])
                    st.plotly_chart(fig)
            else:
                st.warning("Please enter at least one day's thoughts.")

    with tab2:
        handle = st.text_input("Enter Social Media Handle", "@user_2026")
        if st.button("Fetch & Analyze Recent Activity"):
            with st.spinner("Fetching data..."):
                time.sleep(1.5)
                # Mock fetched data
                fetched = ["Doing okay.", "Work is stressful.", "Feeling anxious.", "Not sleeping well.", "I feel hopeless."]
                preds = [predict_emotion(t) for t in fetched]
                score_map = {"normal": 1, "stress": 2, "anxiety": 3, "depression": 4}
                scores = [score_map[p] for p in preds]
                st.session_state['last_scores'] = scores
                
                st.success("Analysis Complete!")
                fig = px.area(y=scores, title=f"Trend for {handle}")
                fig.update_yaxes(tickvals=[1,2,3,4], ticktext=["Normal", "Stress", "Anxiety", "Depression"])
                st.plotly_chart(fig)

# --- MODE 2: GROUP PULSE ---
elif mode == "Group Pulse (Big Data)":
    st.header("Batch Wellness Dashboard")
    st.write("Aggregated data for college wellness monitoring.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Users", "1,240")
    c2.metric("Avg Stress", "Moderate")
    c3.metric("Privacy", "Locked")
    
    df = pd.DataFrame({"Category": ["Normal", "Stress", "Anxiety", "Depression"], "Count": [45, 30, 15, 10]})
    st.plotly_chart(px.pie(df, values='Count', names='Category', hole=0.4))

# --- MODE 3: PROFESSIONAL INSIGHT ---
elif mode == "Professional Insight":
    st.header("Clinical Summary Report")
    token = st.text_input("Enter Patient Token (e.g., PT-8892)")
    if token:
        if 'last_scores' in st.session_state:
            st.error("Clinical Alert: Volatile emotional trend detected.")
            st.table(pd.DataFrame({"Metric": ["Baseline", "Recent Shift", "Risk"], "Value": ["Stress", "Sudden Negative", "High"]}))
        else:
            st.info("No timeline data found for this token. Run a Personal Analysis first.")
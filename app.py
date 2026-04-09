import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os
from pathlib import Path

from explainability import predict_with_explanation
from social_media import (
    SocialFetchError,
    fetch_recent_posts,
    get_platform_descriptor,
    get_platform_descriptors,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model_v2.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer_v2.pkl"
BATCH_REPORT_PATH = BASE_DIR / "data" / "batch_wellness_report.csv"
SCORE_MAP = {"normal": 1, "stress": 2, "anxiety": 3, "depression": 4}
SCORE_LABELS = {1: "Normal", 2: "Stress", 3: "Anxiety", 4: "Depression"}

# --- 1. SETTINGS & ASSETS ---
st.set_page_config(page_title="EmoTrack: AI Mental Health Tracker", layout="wide", page_icon="🧠")

@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with VECTORIZER_PATH.open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

load_error = None
try:
    model, vectorizer = load_model()
except Exception as exc:
    load_error = exc

if load_error is not None:
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        st.error(
            "Model files not found.\n"
            f"Expected:\n- {MODEL_PATH}\n- {VECTORIZER_PATH}"
        )
    elif isinstance(load_error, ModuleNotFoundError):
        st.error(
            "The model files exist, but the current Python environment is missing a required package: "
            f"`{load_error.name}`.\n\n"
            "Start the app with the project virtual environment:\n"
            r"`.\.venv\Scripts\streamlit run app.py --server.port 8502`"
        )
    else:
        st.error(f"Model files exist, but loading failed: {load_error}")
    st.stop()

@st.cache_data
def load_batch_report():
    if not BATCH_REPORT_PATH.exists():
        return None
    return pd.read_csv(BATCH_REPORT_PATH)


def normalize_batch_report(dataframe):
    df = dataframe.copy()
    df.columns = [column.strip().lower() for column in df.columns]
    rename_map = {
        "state": "last_state",
        "status": "last_state",
        "latest_state": "last_state",
        "score": "avg_score",
        "average_score": "avg_score",
    }
    df = df.rename(columns=rename_map)

    required_columns = {"student_id", "department", "last_state", "avg_score"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Batch data must include these columns: "
            + ", ".join(sorted(required_columns))
        )

    df["student_id"] = df["student_id"].astype(str)
    df["department"] = df["department"].astype(str).str.strip()
    df["last_state"] = df["last_state"].astype(str).str.lower().str.strip()
    df["avg_score"] = pd.to_numeric(df["avg_score"], errors="coerce")
    df = df.dropna(subset=["student_id", "department", "last_state", "avg_score"])
    df = df[df["last_state"].isin(SCORE_MAP)]
    df["risk_band"] = df["avg_score"].apply(get_risk_band_from_score)
    return df


# --- 2. CORE LOGIC FUNCTIONS ---
def analyze_entries(entries):
    explanations = [
        predict_with_explanation(text, model, vectorizer)
        for text in entries
    ]
    scores = [SCORE_MAP[item["label"]] for item in explanations]
    return explanations, scores


def get_risk_band_from_score(value):
    if value >= 3.5:
        return "Critical Risk"
    if value >= 2.5:
        return "High Risk"
    if value >= 1.8:
        return "Medium Risk"
    return "Low Risk"


def get_trend_label(scores):
    if not scores:
        return "No Data"
    if all(score == scores[0] for score in scores):
        return "Stable"
    if scores == sorted(scores) and len(set(scores)) > 1:
        return "Worsening Mental State"
    if scores == sorted(scores, reverse=True) and len(set(scores)) > 1:
        return "Improving"
    return "Fluctuating"


def get_shift_label(scores):
    if len(scores) < 2:
        return "No Major Shift"
    if scores[-2] >= 3 and scores[-1] == 1:
        return "Sudden Positive Shift"
    if scores[-2] == 1 and scores[-1] >= 3:
        return "Sudden Negative Shift"
    if abs(scores[-1] - scores[-2]) >= 2:
        return "Abrupt State Change"
    return "No Major Shift"


def summarize_case(explanations, scores):
    avg_score = sum(scores) / len(scores)
    risk, suggestion = get_risk_level(scores)
    severe_entries = sum(
        1 for item in explanations
        if item["label"] in {"anxiety", "depression"}
    )
    return {
        "risk": risk,
        "suggestion": suggestion,
        "trend": get_trend_label(scores),
        "shift": get_shift_label(scores),
        "avg_score": avg_score,
        "latest_state": explanations[-1]["label"].title(),
        "avg_confidence": sum(item["confidence"] for item in explanations) / len(explanations),
        "severe_entries": severe_entries,
    }


def store_latest_analysis(entries, labels, explanations, scores, source="manual"):
    st.session_state["last_entries"] = entries
    st.session_state["last_labels"] = labels
    st.session_state["last_analysis"] = explanations
    st.session_state["last_scores"] = scores
    st.session_state["last_source"] = source


def build_clinical_summary(case_name, labels, explanations, scores):
    case = summarize_case(explanations, scores)
    latest_keywords = explanations[-1]["keywords"][:3]
    repeated_states = pd.Series([item["label"] for item in explanations]).value_counts()
    dominant_state = repeated_states.index[0].title()
    return (
        f"{case_name} shows a {case['trend'].lower()} pattern across {len(scores)} observations, "
        f"with current risk assessed as {case['risk'].lower()} and the latest state classified as "
        f"{case['latest_state'].lower()}. The dominant state across the timeline is {dominant_state.lower()}, "
        f"and {case['severe_entries']} entries were classified as anxiety or depression. "
        f"The most recent explanatory keywords were "
        f"{', '.join(latest_keywords) if latest_keywords else 'not strongly defined'}."
    )


def build_flagged_entries_table(labels, explanations, scores):
    flagged = []
    for label, explanation, score in zip(labels, explanations, scores):
        if score >= 3:
            flagged.append(
                {
                    "Entry": label,
                    "State": explanation["label"].title(),
                    "Confidence": f"{explanation['confidence'] * 100:.1f}%",
                    "Keywords": ", ".join(explanation["keywords"]) or "No strong keywords",
                }
            )
    return pd.DataFrame(flagged)


def render_case_overview(case_name, labels, explanations, scores):
    case = summarize_case(explanations, scores)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk", case["risk"])
    col2.metric("Trend", case["trend"])
    col3.metric("Latest State", case["latest_state"])
    col4.metric("Avg Confidence", f"{case['avg_confidence'] * 100:.1f}%")

    st.info(f"Shift: {case['shift']}  |  Suggestion: {case['suggestion']}")
    render_score_chart(scores, title=f"{case_name} timeline", x_labels=labels)

    summary_text = build_clinical_summary(case_name, labels, explanations, scores)
    st.subheader("Clinical Summary")
    st.write(summary_text)

    flagged_entries = build_flagged_entries_table(labels, explanations, scores)
    if not flagged_entries.empty:
        st.subheader("Flagged Entries")
        st.dataframe(flagged_entries, use_container_width=True)
    else:
        st.success("No anxiety or depression entries were flagged in this timeline.")


def build_case_export(labels, entries, explanations, scores):
    return pd.DataFrame(
        {
            "entry_label": labels,
            "text": entries,
            "predicted_state": [item["label"] for item in explanations],
            "confidence": [round(item["confidence"], 4) for item in explanations],
            "keywords": [", ".join(item["keywords"]) for item in explanations],
            "score": scores,
        }
    )


def render_entry_explanations(entries, explanations, label_prefix, entry_labels=None):
    labels = entry_labels or [f"{label_prefix} {index}" for index in range(1, len(entries) + 1)]
    summary = pd.DataFrame(
        {
            "Entry": labels,
            "Prediction": [item["label"].title() for item in explanations],
            "Confidence": [f"{item['confidence'] * 100:.1f}%" for item in explanations],
            "Keywords": [
                ", ".join(item["keywords"]) if item["keywords"] else "No strong keywords"
                for item in explanations
            ],
        }
    )

    st.subheader("Prediction Details")
    st.caption("Confidence is derived from normalized model scores and is best used comparatively.")
    st.dataframe(summary, use_container_width=True)

    for index, explanation in enumerate(explanations, start=1):
        confidence_text = f"{explanation['confidence'] * 100:.1f}% confidence"
        title = f"{labels[index - 1]}: {explanation['label'].title()} ({confidence_text})"

        with st.expander(title, expanded=index == 1):
            if explanation["source"] == "rule":
                st.info("A safety keyword rule reinforced this prediction.")
            st.caption(
                "Keywords: "
                + (
                    ", ".join(explanation["keywords"])
                    if explanation["keywords"]
                    else "No strong keywords"
                )
            )
            st.markdown(explanation["highlighted_text"], unsafe_allow_html=True)


def render_social_posts(posts):
    summary = pd.DataFrame(
        {
            "Platform": [post.platform for post in posts],
            "Author": [post.author for post in posts],
            "Created At": [post.created_at or "Unknown" for post in posts],
            "Preview": [
                (post.text[:140] + "...")
                if len(post.text) > 140
                else post.text
                for post in posts
            ],
        }
    )

    st.subheader("Fetched Posts")
    st.dataframe(summary, use_container_width=True)

    for index, post in enumerate(posts, start=1):
        title = f"Post {index} from {post.author}"
        with st.expander(title, expanded=index == 1):
            if post.created_at:
                st.caption(f"Created: {post.created_at}")
            st.write(post.text)
            if post.url:
                st.markdown(f"[Open original post]({post.url})")


def render_score_chart(scores, title, x_labels=None):
    x_values = x_labels or [f"Point {index}" for index in range(1, len(scores) + 1)]
    fig = px.area(y=scores, x=x_values, title=title)
    fig.update_yaxes(tickvals=[1, 2, 3, 4], ticktext=["Normal", "Stress", "Anxiety", "Depression"])
    st.plotly_chart(fig)


def get_social_auth():
    try:
        x_bearer_token = st.secrets.get("X_BEARER_TOKEN")
    except Exception:
        x_bearer_token = None

    return {
        "x_bearer_token": x_bearer_token,
    }


def has_x_auth():
    auth = get_social_auth()
    return bool(auth.get("x_bearer_token") or os.getenv("X_BEARER_TOKEN"))


def get_risk_level(scores):
    avg = sum(scores) / len(scores)
    if avg >= 3.5: return "Critical Risk", "Immediate support recommended."
    elif avg >= 2.5: return "High Risk", "Consider professional guidance."
    elif avg >= 1.8: return "Medium Risk", "Monitor trends and take a break."
    return "Low Risk", "Your emotional trend appears stable."

# --- 3. UI SIDEBAR ---
st.sidebar.title("Choose Mode")
mode = st.sidebar.selectbox(
    "Mode",
    ["Personal Analysis", "Group Pulse (Big Data)", "Professional Insight"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("Project by: Group of Two")

# --- 4. MAIN INTERFACE ---

# --- MODE 1: PERSONAL ANALYSIS ---
if mode == "Personal Analysis":
    st.header("Personal Timeline Tracker")
    st.write("Enter your thoughts from the last few days to see your emotional trend.")

    tab1, tab2 = st.tabs(["Manual Entry", "🔗 Auto-Fetch Social Media"])

    with tab1:
        timeline_length = st.slider(
            "Timeline length",
            min_value=5,
            max_value=30,
            value=10,
            help="Increase this to analyze a longer manual timeline.",
        )

        manual_entries = []
        manual_labels = []

        for i in range(1, timeline_length + 1):
            with st.expander(f"Timeline Entry {i}", expanded=i <= 3):
                text_value = st.text_area(
                    f"Entry {i} text",
                    key=f"d{i}_text",
                    height=100,
                    placeholder="Write a journal entry or post text...",
                )
                analyzable_text = text_value.strip()
                if analyzable_text:
                    manual_entries.append(analyzable_text)
                    manual_labels.append(f"Entry {i}")

        if st.button("Analyze My Timeline"):
            if manual_entries:
                explanations, scores = analyze_entries(manual_entries)
                store_latest_analysis(
                    manual_entries,
                    manual_labels,
                    explanations,
                    scores,
                    source="manual",
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    risk, sug = get_risk_level(scores)
                    st.metric("Risk Status", risk)
                    st.info(f"**Suggestion:** {sug}")
                with col2:
                    fig = px.line(y=scores, x=manual_labels, markers=True)
                    fig.update_yaxes(tickvals=[1,2,3,4], ticktext=["Normal", "Stress", "Anxiety", "Depression"])
                    st.plotly_chart(fig)

                render_entry_explanations(
                    manual_entries,
                    explanations,
                    "Entry",
                    entry_labels=manual_labels,
                )
            else:
                st.warning("Please enter at least one timeline item.")

    with tab2:
        platform_options = {
            descriptor.label: descriptor.key
            for descriptor in get_platform_descriptors()
        }
        selected_platform = st.selectbox("Platform", list(platform_options.keys()))
        platform_key = platform_options[selected_platform]
        descriptor = get_platform_descriptor(platform_key)

        st.caption(descriptor.handle_help)
        st.caption(f"Setup: {descriptor.auth_hint}")

        if platform_key == "x" and not has_x_auth():
            st.warning("X is selected, but `X_BEARER_TOKEN` is not configured yet.")

        handle = st.text_input(
            "Account / Handle",
            placeholder=descriptor.placeholder,
            key=f"{platform_key}_handle",
        )

        instance_url = ""
        if descriptor.needs_instance:
            instance_url = st.text_input(
                "Instance URL",
                placeholder=descriptor.instance_placeholder,
                help="Optional when the handle already includes the server domain.",
                key=f"{platform_key}_instance",
            )

        post_limit = st.slider(
            "Recent posts to analyze",
            min_value=3,
            max_value=20,
            value=5,
            key=f"{platform_key}_limit",
        )

        fetch_all_history = st.checkbox(
            "Fetch full available account history",
            key=f"{platform_key}_fetch_all",
            help=(
                "Pages through the platform API until older posts run out. "
                "Private, deleted, or API-restricted posts still cannot be fetched."
            ),
        )
        max_history_posts = None
        if fetch_all_history:
            history_cap_input = st.number_input(
                "Optional safety cap on total fetched posts",
                min_value=0,
                max_value=100000,
                value=0,
                step=100,
                help="Set 0 for no explicit cap. Use a cap if a profile has a very long history.",
                key=f"{platform_key}_history_cap",
            )
            max_history_posts = None if history_cap_input == 0 else int(history_cap_input)
            st.caption(
                "Full-history fetches can take longer on older accounts and may still be limited by each platform API."
            )
            if platform_key == "x":
                st.caption("X history availability depends on your current API access and platform-side restrictions.")

        if st.button("Fetch & Analyze Recent Activity"):
            if not handle.strip():
                st.warning("Please enter a social media handle first.")
            else:
                fetch_scope = "full account history" if fetch_all_history else "recent posts"
                with st.spinner(f"Fetching {fetch_scope} from {selected_platform}..."):
                    try:
                        posts = fetch_recent_posts(
                            platform_key,
                            handle,
                            post_limit,
                            instance_url=instance_url or None,
                            auth=get_social_auth(),
                            fetch_all=fetch_all_history,
                            max_posts=max_history_posts,
                        )
                    except SocialFetchError as exc:
                        st.error(str(exc))
                    else:
                        fetched = [post.text for post in posts]
                        post_labels = [f"Post {index}" for index in range(1, len(posts) + 1)]
                        explanations, scores = analyze_entries(fetched)
                        store_latest_analysis(
                            fetched,
                            post_labels,
                            explanations,
                            scores,
                            source=platform_key,
                        )
                        st.session_state['last_social_posts'] = [post.to_dict() for post in posts]
                        
                        st.success(f"Fetched {len(posts)} posts from {selected_platform}.")
                        render_score_chart(
                            scores,
                            title=f"{selected_platform} trend for {handle}",
                            x_labels=post_labels,
                        )
                        render_social_posts(posts)
                        render_entry_explanations(
                            fetched,
                            explanations,
                            "Post",
                            entry_labels=post_labels,
                        )

# --- MODE 2: GROUP PULSE ---
elif mode == "Group Pulse (Big Data)":
    st.header("Batch Wellness Dashboard")
    st.write("Analyze a cohort-level wellness file to see risk distribution, department hotspots, and students who may need follow-up.")

    uploaded_batch = st.file_uploader(
        "Upload batch wellness CSV",
        type=["csv"],
        help="Expected columns: student_id, department, last_state, avg_score",
        key="group_batch_upload",
    )

    batch_source = "sample batch_wellness_report.csv"
    raw_batch_df = None
    if uploaded_batch is not None:
        raw_batch_df = pd.read_csv(uploaded_batch)
        batch_source = "uploaded CSV"
    else:
        raw_batch_df = load_batch_report()

    if raw_batch_df is None:
        st.warning("No batch report found yet. Upload a CSV or generate `data/batch_wellness_report.csv` first.")
    else:
        try:
            batch_df = normalize_batch_report(raw_batch_df)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.caption(f"Data source: {batch_source}")

            filter_col1, filter_col2, filter_col3 = st.columns(3)
            departments = sorted(batch_df["department"].unique())
            states = sorted(batch_df["last_state"].unique())
            risk_bands = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]

            selected_departments = filter_col1.multiselect(
                "Departments",
                departments,
                default=departments,
            )
            selected_states = filter_col2.multiselect(
                "Last states",
                states,
                default=states,
            )
            selected_risks = filter_col3.multiselect(
                "Risk bands",
                risk_bands,
                default=risk_bands,
            )

            filtered_df = batch_df[
                batch_df["department"].isin(selected_departments)
                & batch_df["last_state"].isin(selected_states)
                & batch_df["risk_band"].isin(selected_risks)
            ].copy()

            if filtered_df.empty:
                st.warning("No students match the current filters.")
            else:
                high_risk_mask = filtered_df["risk_band"].isin(["High Risk", "Critical Risk"])
                dominant_state = filtered_df["last_state"].mode().iloc[0].title()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Students", f"{filtered_df['student_id'].nunique():,}")
                c2.metric("Avg Score", f"{filtered_df['avg_score'].mean():.2f}")
                c3.metric("High Risk Share", f"{high_risk_mask.mean() * 100:.1f}%")
                c4.metric("Dominant State", dominant_state)

                state_counts = (
                    filtered_df["last_state"]
                    .value_counts()
                    .rename_axis("State")
                    .reset_index(name="Count")
                )
                state_counts["State"] = state_counts["State"].str.title()

                dept_summary = (
                    filtered_df.groupby("department", as_index=False)
                    .agg(
                        students=("student_id", "nunique"),
                        avg_score=("avg_score", "mean"),
                    )
                )
                dept_high_risk = (
                    filtered_df.assign(high_risk=high_risk_mask)
                    .groupby("department")["high_risk"]
                    .mean()
                    .mul(100)
                    .reset_index(name="high_risk_pct")
                )
                dept_summary = dept_summary.merge(dept_high_risk, on="department", how="left")
                dept_summary = dept_summary.sort_values("avg_score", ascending=False)

                risk_by_department = (
                    filtered_df.groupby(["department", "risk_band"])
                    .size()
                    .reset_index(name="Count")
                )

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.plotly_chart(
                        px.pie(state_counts, values="Count", names="State", hole=0.4, title="State Distribution"),
                        use_container_width=True,
                    )
                with chart_col2:
                    st.plotly_chart(
                        px.bar(
                            dept_summary,
                            x="department",
                            y="avg_score",
                            color="high_risk_pct",
                            title="Average Score by Department",
                            labels={"department": "Department", "avg_score": "Average Score", "high_risk_pct": "High Risk %"},
                        ),
                        use_container_width=True,
                    )

                chart_col3, chart_col4 = st.columns(2)
                with chart_col3:
                    st.plotly_chart(
                        px.bar(
                            risk_by_department,
                            x="department",
                            y="Count",
                            color="risk_band",
                            title="Risk Bands by Department",
                            barmode="stack",
                        ),
                        use_container_width=True,
                    )
                with chart_col4:
                    st.plotly_chart(
                        px.histogram(
                            filtered_df,
                            x="avg_score",
                            nbins=20,
                            color="last_state",
                            title="Score Distribution",
                        ),
                        use_container_width=True,
                    )

                st.subheader("Departments Needing Attention")
                attention_table = dept_summary.rename(
                    columns={
                        "department": "Department",
                        "students": "Students",
                        "avg_score": "Average Score",
                        "high_risk_pct": "High Risk %",
                    }
                )
                attention_table["Average Score"] = attention_table["Average Score"].round(2)
                attention_table["High Risk %"] = attention_table["High Risk %"].round(1)
                st.dataframe(attention_table, use_container_width=True)

                flagged_students = filtered_df[
                    filtered_df["risk_band"].isin(["High Risk", "Critical Risk"])
                    | filtered_df["last_state"].isin(["anxiety", "depression"])
                ].sort_values(["avg_score", "last_state"], ascending=[False, True])
                flagged_students = flagged_students.copy()
                flagged_students["last_state"] = flagged_students["last_state"].str.title()

                st.subheader("Students Requiring Follow-up")
                st.dataframe(
                    flagged_students.rename(
                        columns={
                            "student_id": "Student ID",
                            "department": "Department",
                            "last_state": "Last State",
                            "avg_score": "Average Score",
                            "risk_band": "Risk Band",
                        }
                    ),
                    use_container_width=True,
                )

                st.download_button(
                    "Download filtered cohort CSV",
                    filtered_df.to_csv(index=False).encode("utf-8"),
                    file_name="filtered_batch_wellness_report.csv",
                    mime="text/csv",
                )

# --- MODE 3: PROFESSIONAL INSIGHT ---
elif mode == "Professional Insight":
    st.header("Clinical Summary Report")
    st.write("Review a patient timeline from the latest app analysis, pasted notes, or an uploaded CSV.")

    case_name = st.text_input("Patient / Case ID", "PT-8892")
    safe_case_name = (case_name.strip() or "patient_case").lower().replace(" ", "_")
    tab_latest, tab_custom = st.tabs(["Use Latest Analysis", "Paste or Upload Timeline"])

    with tab_latest:
        if "last_analysis" not in st.session_state or "last_scores" not in st.session_state:
            st.info("No recent analysis is available yet. Run a Personal Analysis first, or use the upload tab.")
        else:
            latest_entries = st.session_state.get("last_entries", [])
            latest_labels = st.session_state.get(
                "last_labels",
                [f"Entry {index}" for index in range(1, len(st.session_state["last_scores"]) + 1)],
            )
            latest_source = st.session_state.get("last_source", "manual")

            st.caption(f"Source: {latest_source}")
            render_case_overview(
                case_name,
                latest_labels,
                st.session_state["last_analysis"],
                st.session_state["last_scores"],
            )
            render_entry_explanations(
                latest_entries,
                st.session_state["last_analysis"],
                "Entry",
                entry_labels=latest_labels,
            )

            latest_export = build_case_export(
                latest_labels,
                latest_entries,
                st.session_state["last_analysis"],
                st.session_state["last_scores"],
            )
            st.download_button(
                "Download clinical report CSV",
                latest_export.to_csv(index=False).encode("utf-8"),
                file_name=f"{safe_case_name}_clinical_report.csv",
                mime="text/csv",
            )

    with tab_custom:
        input_mode = st.radio(
            "Timeline input",
            ["Paste notes", "Upload CSV"],
            horizontal=True,
        )

        custom_entries = []
        custom_labels = []

        if input_mode == "Paste notes":
            raw_notes = st.text_area(
                "Patient timeline notes",
                height=220,
                placeholder="Enter one observation per line...",
            )
            if raw_notes.strip():
                custom_entries = [line.strip() for line in raw_notes.splitlines() if line.strip()]
                custom_labels = [f"Observation {index}" for index in range(1, len(custom_entries) + 1)]
        else:
            uploaded_timeline = st.file_uploader(
                "Upload patient timeline CSV",
                type=["csv"],
                key="professional_timeline_csv",
            )
            if uploaded_timeline is not None:
                timeline_df = pd.read_csv(uploaded_timeline)
                st.dataframe(timeline_df.head(10), use_container_width=True)
                columns = list(timeline_df.columns)
                text_column = st.selectbox("Text column", columns, key="professional_text_column")
                label_options = ["(auto labels)"] + columns
                label_column = st.selectbox("Label column", label_options, key="professional_label_column")
                extra_text_column = st.selectbox(
                    "Optional second text column",
                    ["(none)"] + columns,
                    key="professional_extra_text_column",
                )

                combined_text = timeline_df[text_column].fillna("").astype(str)
                if extra_text_column != "(none)":
                    combined_text = combined_text.str.cat(
                        timeline_df[extra_text_column].fillna("").astype(str),
                        sep="\n",
                    )

                custom_entries = [value.strip() for value in combined_text.tolist() if value.strip()]
                if label_column == "(auto labels)":
                    custom_labels = [f"Observation {index}" for index in range(1, len(custom_entries) + 1)]
                else:
                    candidate_labels = timeline_df[label_column].fillna("").astype(str).tolist()
                    paired = [
                        (label_value.strip() or f"Observation {index}", text_value.strip())
                        for index, (label_value, text_value) in enumerate(zip(candidate_labels, combined_text.tolist()), start=1)
                        if text_value.strip()
                    ]
                    custom_labels = [item[0] for item in paired]
                    custom_entries = [item[1] for item in paired]

        if st.button("Generate Professional Report"):
            if not custom_entries:
                st.warning("Please provide patient notes or upload a valid timeline CSV first.")
            else:
                custom_explanations, custom_scores = analyze_entries(custom_entries)
                render_case_overview(case_name, custom_labels, custom_explanations, custom_scores)
                render_entry_explanations(
                    custom_entries,
                    custom_explanations,
                    "Observation",
                    entry_labels=custom_labels,
                )

                custom_export = build_case_export(
                    custom_labels,
                    custom_entries,
                    custom_explanations,
                    custom_scores,
                )
                st.download_button(
                    "Download generated report CSV",
                    custom_export.to_csv(index=False).encode("utf-8"),
                    file_name=f"{safe_case_name}_uploaded_report.csv",
                    mime="text/csv",
                )

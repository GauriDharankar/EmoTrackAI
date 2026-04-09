import html
import math
import re


RULE_KEYWORDS = {
    "depression": ["hopeless", "worthless", "suicide", "empty", "tired"],
    "anxiety": ["panic", "shaking", "nervous", "anxious", "racing"],
}

HIGHLIGHT_STYLE = (
    "background-color: #fde68a; color: #111827; "
    "padding: 0.08rem 0.24rem; border-radius: 0.25rem;"
)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_keyword(text, keyword):
    pattern = re.compile(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)")
    return bool(pattern.search(text))


def find_rule_match(text_clean):
    for label, keywords in RULE_KEYWORDS.items():
        matched = [keyword for keyword in keywords if _contains_keyword(text_clean, keyword)]
        if matched:
            return label, matched
    return None, []


def _softmax(values):
    if not values:
        return []

    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def get_confidence_scores(model, vec):
    classes = list(model.classes_)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vec)[0]
        return {
            label: float(probability)
            for label, probability in zip(classes, probabilities)
        }

    raw_scores = model.decision_function(vec)

    if getattr(raw_scores, "ndim", 1) == 1:
        margin = float(raw_scores[0])
        positive_probability = 1.0 / (1.0 + math.exp(-margin))
        return {
            classes[0]: 1.0 - positive_probability,
            classes[1]: positive_probability,
        }

    probabilities = _softmax([float(score) for score in raw_scores[0]])
    return {
        label: probability
        for label, probability in zip(classes, probabilities)
    }


def get_top_keywords(model, vectorizer, vec, label, top_n=5):
    feature_names = vectorizer.get_feature_names_out()

    if vec.nnz == 0:
        return []

    indices = vec.indices
    values = vec.data

    if hasattr(model, "coef_"):
        classes = list(model.classes_)
        class_index = classes.index(label)
        coefficients = model.coef_

        if coefficients.shape[0] == 1 and len(classes) == 2:
            class_weights = coefficients[0]
            if label == classes[0]:
                contributions = values * (-class_weights[indices])
            else:
                contributions = values * class_weights[indices]
        else:
            contributions = values * coefficients[class_index, indices]

        ranked = sorted(
            zip(indices, contributions),
            key=lambda item: item[1],
            reverse=True,
        )
        keywords = [
            feature_names[index]
            for index, contribution in ranked
            if contribution > 0
        ]

        if keywords:
            return list(dict.fromkeys(keywords))[:top_n]

    fallback_ranked = sorted(
        zip(indices, values),
        key=lambda item: item[1],
        reverse=True,
    )
    fallback_keywords = [feature_names[index] for index, _ in fallback_ranked]
    return list(dict.fromkeys(fallback_keywords))[:top_n]


def highlight_keywords(text, keywords):
    if not text:
        return ""

    cleaned_keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
    if not cleaned_keywords:
        return html.escape(text)

    unique_keywords = sorted(
        set(cleaned_keywords),
        key=len,
        reverse=True,
    )
    pattern = re.compile(
        r"(?i)(?<!\w)(" + "|".join(re.escape(keyword) for keyword in unique_keywords) + r")(?!\w)"
    )

    highlighted_parts = []
    last_index = 0

    for match in pattern.finditer(text):
        start, end = match.span()
        highlighted_parts.append(html.escape(text[last_index:start]))
        highlighted_parts.append(
            f"<span style=\"{HIGHLIGHT_STYLE}\">{html.escape(text[start:end])}</span>"
        )
        last_index = end

    highlighted_parts.append(html.escape(text[last_index:]))
    return "".join(highlighted_parts)


def predict_with_explanation(text, model, vectorizer, top_n=5):
    text_clean = clean_text(text)
    vec = vectorizer.transform([text_clean])

    model_label = model.predict(vec)[0]
    confidence_scores = get_confidence_scores(model, vec)
    rule_label, rule_keywords = find_rule_match(text_clean)

    final_label = rule_label or model_label
    confidence = confidence_scores.get(final_label, 0.0)

    if rule_label:
        confidence = max(confidence, 0.92)

    model_keywords = get_top_keywords(model, vectorizer, vec, final_label, top_n=top_n)
    keywords = list(dict.fromkeys(rule_keywords + model_keywords))[:top_n]

    return {
        "label": final_label,
        "confidence": float(confidence),
        "keywords": keywords,
        "highlighted_text": highlight_keywords(text, keywords),
        "source": "rule" if rule_label else "model",
        "scores": confidence_scores,
    }

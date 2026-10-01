import streamlit as st
import pandas as pd
import joblib
import re
import string
import html

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Disaster Tweet Classification",
    page_icon="🚨",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f14;
    }

    .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    .dashboard-title {
        font-size: 38px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .dashboard-subtitle {
        font-size: 17px;
        color: #8fa3bf;
        margin-bottom: 25px;
    }

    .metric-card {
        background: #141b24;
        border: 1px solid #263242;
        border-radius: 14px;
        padding: 22px;
        min-height: 130px;
    }

    .metric-title {
        color: #8091aa;
        font-size: 14px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 32px;
        font-weight: 800;
    }

    .section-title {
        font-size: 24px;
        font-weight: 800;
        color: white;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    .tweet-card {
        background: #141b24;
        border: 1px solid #293546;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .critical-card {
        border-left: 4px solid #ff5964;
    }

    .high-card {
        border-left: 4px solid #ffb020;
    }

    .medium-card {
        border-left: 4px solid #35c2b1;
    }

    .low-card {
        border-left: 4px solid #7188ff;
    }

    .normal-card {
        border-left: 4px solid #566273;
    }

    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        margin-right: 8px;
    }

    .badge-critical {
        background: rgba(255, 89, 100, 0.15);
        color: #ff5964;
    }

    .badge-high {
        background: rgba(255, 176, 32, 0.15);
        color: #ffb020;
    }

    .badge-medium {
        background: rgba(53, 194, 177, 0.15);
        color: #35c2b1;
    }

    .badge-low {
        background: rgba(113, 136, 255, 0.15);
        color: #8ea1ff;
    }

    .badge-na {
        background: #1c2633;
        color: #8293aa;
    }

    .badge-assistance {
        background: #1c2633;
        color: #aebbd0;
    }

    .tweet-text {
        color: #f3f6fa;
        font-size: 17px;
        line-height: 1.6;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .tweet-meta {
        color: #8293aa;
        font-size: 14px;
    }

    .tweet-meta b {
        color: #ffffff;
    }

    .feed-info {
        color: #8fa3bf;
        font-size: 15px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# INITIALIZE NLP
# =========================================================

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


# =========================================================
# TEXT PREPROCESSING FOR MODEL
# =========================================================

def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol
    text = re.sub(r"#", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# =========================================================
# TEXT NORMALIZATION FOR DUPLICATE REMOVAL
# =========================================================

def normalize_for_duplicate_check(text):
    """
    Normalize tweets for duplicate detection.
    Removes URLs, mentions, punctuation, numbers and extra spaces.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove hashtag symbol
    text = text.replace("#", " ")

    # Remove numbers and special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# REMOVE EXACT AND NEAR DUPLICATES
# =========================================================

def remove_duplicate_tweets(dataframe):

    dataframe = dataframe.copy()

    # Create normalized text
    dataframe["duplicate_text"] = (
        dataframe["text"]
        .fillna("")
        .astype(str)
        .apply(normalize_for_duplicate_check)
    )

    # Remove empty tweets
    dataframe = dataframe[
        dataframe["duplicate_text"].ne("")
    ].reset_index(drop=True)

    # -----------------------------------------------------
    # STEP 1: REMOVE EXACT DUPLICATES
    # -----------------------------------------------------

    dataframe = dataframe.drop_duplicates(
        subset=["duplicate_text"],
        keep="first"
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # STEP 2: REMOVE NEAR DUPLICATES
    # -----------------------------------------------------

    if len(dataframe) > 1:

        similarity_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000
        )

        similarity_matrix = similarity_vectorizer.fit_transform(
            dataframe["duplicate_text"]
        )

        # Cosine distance threshold
        # 0.15 approximately means 85% similarity

        duplicate_distance = 0.15

        neighbor_model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            radius=duplicate_distance,
            n_jobs=-1
        )

        neighbor_model.fit(similarity_matrix)

        distances, neighbors = neighbor_model.radius_neighbors(
            similarity_matrix,
            return_distance=True
        )

        rows_to_remove = set()

        # Keep the first tweet and remove later similar tweets
        for current_position in range(len(dataframe)):

            if current_position in rows_to_remove:
                continue

            for neighbor_position, distance in zip(
                neighbors[current_position],
                distances[current_position]
            ):

                # Ignore the tweet itself
                if neighbor_position == current_position:
                    continue

                # Remove only later rows
                if neighbor_position > current_position:

                    if distance <= duplicate_distance:
                        rows_to_remove.add(neighbor_position)

        # Remove duplicate rows
        if rows_to_remove:
            dataframe = dataframe.drop(
                index=list(rows_to_remove)
            ).reset_index(drop=True)

    # Remove helper column
    dataframe = dataframe.drop(
        columns=["duplicate_text"],
        errors="ignore"
    )

    return dataframe


# =========================================================
# LOAD DATASET
# =========================================================

try:
    data = pd.read_csv("dataset/train.csv")

except FileNotFoundError:
    st.error(
        "Dataset not found. Please check that "
        "'dataset/train.csv' exists."
    )
    st.stop()


if "text" not in data.columns:
    st.error(
        "The dataset must contain a column named 'text'."
    )
    st.stop()


# Remove duplicates silently
data = remove_duplicate_tweets(data)


# =========================================================
# LOAD TRAINED MODEL AND VECTORIZER
# =========================================================

try:
    model = joblib.load(
        "models/disaster_model.pkl"
    )

    vectorizer = joblib.load(
        "models/vectorizer.pkl"
    )

except FileNotFoundError:
    st.error(
        "Model files not found. Please check that these files exist:\n\n"
        "- models/disaster_model.pkl\n"
        "- models/vectorizer.pkl"
    )
    st.stop()


# =========================================================
# MODEL PREDICTION
# =========================================================

cleaned_text = data["text"].apply(clean_text)

X = vectorizer.transform(cleaned_text)

predictions = model.predict(X)

data["prediction"] = predictions


# =========================================================
# MODEL CONFIDENCE
# =========================================================

try:
    probabilities = model.predict_proba(X)

    data["confidence"] = (
        probabilities.max(axis=1) * 100
    )

except AttributeError:
    data["confidence"] = 0.0


# =========================================================
# CLASSIFICATION
# =========================================================

data["classification"] = data["prediction"].apply(
    lambda value: "Disaster"
    if value == 1
    else "Non-Disaster"
)


# =========================================================
# ASSISTANCE TYPE DETECTION
# =========================================================

def detect_assistance(text):

    text = str(text).lower()

    # RESCUE
    rescue_words = [
        "rescue", "rescued", "trapped", "stuck",
        "stranded", "save me", "save us",
        "evacuate", "evacuation", "evacuating",
        "need help", "needs help", "help us",
        "missing", "missing people"
    ]

    if any(word in text for word in rescue_words):
        return "🆘 Rescue"

    # AIRCRAFT / CRASH / ACCIDENT
    accident_words = [
        "plane", "aircraft", "airplane", "flight",
        "plane crash", "aircraft crash", "crash",
        "crashed", "debris", "aircraft debris",
        "plane debris", "missing aircraft",
        "missing plane", "wreckage", "wreck",
        "collision", "aviation accident",
        "air accident"
    ]

    if any(word in text for word in accident_words):
        return "🆘 Rescue"

    # MEDICAL
    medical_words = [
        "injured", "injury", "injuries",
        "hospital", "doctor", "medical",
        "ambulance", "patient", "wounded",
        "bleeding", "paramedic", "hurt",
        "casualties"
    ]

    if any(word in text for word in medical_words):
        return "🩺 Medical Assistance"

    # FOOD AND WATER
    food_words = [
        "food", "water", "hungry", "thirsty",
        "drinking water", "food supplies",
        "water supplies", "clean water",
        "ration", "rations"
    ]

    if any(word in text for word in food_words):
        return "🍱 Food & Water"

    # SHELTER
    shelter_words = [
        "shelter", "homeless", "no home",
        "home destroyed", "homes destroyed",
        "roof destroyed", "need a place",
        "temporary shelter", "shelter in place",
        "refugee camp"
    ]

    if any(word in text for word in shelter_words):
        return "🏠 Shelter"

    # INFRASTRUCTURE
    infrastructure_words = [
        "bridge collapsed", "bridge collapse",
        "road damaged", "road destroyed",
        "road collapse", "building collapsed",
        "building collapse", "building damage",
        "power outage", "electricity",
        "power lines", "infrastructure",
        "damaged bridge", "damaged road",
        "collapsed building", "collapse"
    ]

    if any(word in text for word in infrastructure_words):
        return "🏗️ Infrastructure Damage"

    # GENERAL ASSISTANCE
    general_words = [
        "emergency", "disaster", "crisis",
        "urgent", "emergency response",
        "emergency situation", "disaster response",
        "attack", "bombing", "explosion",
        "terrorist"
    ]

    if any(word in text for word in general_words):
        return "🆘 General Assistance"

    return "🆘 General Assistance"


data["assistance"] = data["text"].apply(
    detect_assistance
)


# =========================================================
# PRIORITY CALCULATION
# =========================================================

def calculate_priority(row):

    # Non-disaster tweets do not receive priority
    if row["prediction"] == 0:
        return "N/A"

    text = str(row["text"]).lower()

    # CRITICAL
    critical_words = [
        "trapped", "stuck", "stranded", "rescue",
        "dying", "dead", "killed", "killing",
        "death", "deaths", "injured", "injury",
        "emergency", "evacuate", "evacuation",
        "urgent", "help us", "save us",
        "missing", "missing people", "collapsed",
        "collapse", "explosion", "bombing",
        "bomb", "suicide bomber", "terrorist attack",
        "mass casualty"
    ]

    if any(word in text for word in critical_words):
        return "Critical"

    # HIGH
    high_words = [
        "flood", "fire", "earthquake", "tornado",
        "hurricane", "cyclone", "wildfire",
        "destroyed", "damage", "crash",
        "plane", "aircraft", "debris", "storm"
    ]

    if any(word in text for word in high_words):
        return "High"

    # MEDIUM
    if row["confidence"] >= 75:
        return "Medium"

    # LOW
    return "Low"


data["priority"] = data.apply(
    calculate_priority,
    axis=1
)


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.markdown(
    """
    <div class="section-title">
        🚨 CLASSIFICATION
    </div>
    """,
    unsafe_allow_html=True
)

classification_filter = st.sidebar.selectbox(
    "Select Classification",
    [
        "All Tweets",
        "Disaster",
        "Non-Disaster"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div class="section-title">
        PRIORITY
    </div>
    """,
    unsafe_allow_html=True
)

priority_filter = st.sidebar.selectbox(
    "Select Priority",
    [
        "All Priorities",
        "Critical",
        "High",
        "Medium",
        "Low"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div class="section-title">
        ASSISTANCE TYPE
    </div>
    """,
    unsafe_allow_html=True
)

assistance_filter = st.sidebar.selectbox(
    "Select Assistance",
    [
        "All Assistance",
        "🆘 Rescue",
        "🩺 Medical Assistance",
        "🍱 Food & Water",
        "🏠 Shelter",
        "🏗️ Infrastructure Damage",
        "🆘 General Assistance"
    ]
)


# =========================================================
# FILTER DATA
# =========================================================

filtered_data = data.copy()

if classification_filter == "Disaster":
    filtered_data = filtered_data[
        filtered_data["classification"] == "Disaster"
    ]

elif classification_filter == "Non-Disaster":
    filtered_data = filtered_data[
        filtered_data["classification"] == "Non-Disaster"
    ]


if priority_filter != "All Priorities":
    filtered_data = filtered_data[
        filtered_data["priority"] == priority_filter
    ]


if assistance_filter != "All Assistance":
    filtered_data = filtered_data[
        filtered_data["assistance"] == assistance_filter
    ]


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="dashboard-title">
        🚨 AI Disaster Tweet Classification & Monitoring
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="dashboard-subtitle">
        AI-powered disaster intelligence dashboard •
        Tweets automatically analyzed from the Kaggle dataset
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# METRICS
# =========================================================

total_tweets = len(data)

disaster_tweets = (
    data["prediction"] == 1
).sum()

non_disaster_tweets = (
    data["prediction"] == 0
).sum()

disaster_data = data[
    data["prediction"] == 1
]

critical_tweets = (
    disaster_data["priority"] == "Critical"
).sum()

average_confidence = data["confidence"].mean()


# =========================================================
# METRIC CARD FUNCTION
# =========================================================

def metric_card_html(title, value):

    return (
        f'<div class="metric-card">'
        f'<div class="metric-title">{title}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>'
    )


# =========================================================
# METRIC CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        metric_card_html(
            "Tweets Analyzed",
            f"{total_tweets:,}"
        ),
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        metric_card_html(
            "Disaster Tweets",
            f"{disaster_tweets:,}"
        ),
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        metric_card_html(
            "Critical Priority",
            f"{critical_tweets:,}"
        ),
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        metric_card_html(
            "Avg. AI Confidence",
            f"{average_confidence:.1f}%"
        ),
        unsafe_allow_html=True
    )


# =========================================================
# CLASSIFICATION SUMMARY
# =========================================================

st.markdown(
    """
    <div class="section-title">
        📊 CLASSIFICATION SUMMARY
    </div>
    """,
    unsafe_allow_html=True
)

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.info(
        f"🚨 Disaster — {disaster_tweets:,}"
    )

with summary_col2:
    st.info(
        f"⚪ Non-Disaster — {non_disaster_tweets:,}"
    )


# =========================================================
# TWEET FEED
# =========================================================

st.markdown(
    """
    <div class="section-title">
        📋 CLASSIFIED TWEET FEED
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SORT BY PRIORITY
# =========================================================

priority_order = {
    "Critical": 1,
    "High": 2,
    "Medium": 3,
    "Low": 4,
    "N/A": 5
}

filtered_data = filtered_data.copy()

filtered_data["priority_order"] = (
    filtered_data["priority"].map(priority_order)
)

sorted_data = filtered_data.sort_values(
    ["priority_order", "confidence"],
    ascending=[True, False]
)


# =========================================================
# LOAD MORE STATE
# =========================================================

filter_key = (
    classification_filter,
    priority_filter,
    assistance_filter
)

if st.session_state.get("last_filter_key") != filter_key:
    st.session_state.show_count = 15
    st.session_state.last_filter_key = filter_key

if "show_count" not in st.session_state:
    st.session_state.show_count = 15


tweets_to_show = sorted_data.head(
    st.session_state.show_count
)


st.markdown(
    f"""
    <div class="feed-info">
        Showing {len(tweets_to_show):,}
        of {len(filtered_data):,} filtered tweets
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DISPLAY TWEETS
# =========================================================

for _, row in tweets_to_show.iterrows():

    priority = row["priority"]

    if priority == "Critical":
        card_class = "critical-card"
        priority_badge = "badge-critical"
        priority_text = "🔴 CRITICAL PRIORITY"

    elif priority == "High":
        card_class = "high-card"
        priority_badge = "badge-high"
        priority_text = "🟠 HIGH PRIORITY"

    elif priority == "Medium":
        card_class = "medium-card"
        priority_badge = "badge-medium"
        priority_text = "🟡 MEDIUM PRIORITY"

    elif priority == "Low":
        card_class = "low-card"
        priority_badge = "badge-low"
        priority_text = "🔵 LOW PRIORITY"

    else:
        card_class = "normal-card"
        priority_badge = "badge-na"
        priority_text = "⚪ PRIORITY N/A"


    if row["prediction"] == 1:
        classification_icon = "🚨"
    else:
        classification_icon = "⚪"


    tweet_text = html.escape(
        str(row["text"])
    )

    assistance_text = html.escape(
        str(row["assistance"])
    )

    classification_text = html.escape(
        str(row["classification"])
    )


    card_html = (
        f'<div class="tweet-card {card_class}">'
        f'<div>'
        f'<span class="badge {priority_badge}">'
        f'{priority_text}'
        f'</span>'
        f'<span class="badge badge-assistance">'
        f'{assistance_text}'
        f'</span>'
        f'</div>'
        f'<div class="tweet-text">'
        f'{classification_icon}'
        f'&nbsp;'
        f'{tweet_text}'
        f'</div>'
        f'<div class="tweet-meta">'
        f'AI Classification: '
        f'<b>{classification_text}</b>'
        f'&nbsp;&nbsp;•&nbsp;&nbsp;'
        f'Confidence: '
        f'<b>{row["confidence"]:.1f}%</b>'
        f'</div>'
        f'</div>'
    )

    st.markdown(
        card_html,
        unsafe_allow_html=True
    )


# =========================================================
# LOAD MORE BUTTON
# =========================================================

if len(tweets_to_show) < len(filtered_data):

    if st.button("Load More Tweets"):
        st.session_state.show_count += 15
        st.rerun()


# =========================================================
# ASSISTANCE BREAKDOWN
# =========================================================

st.markdown(
    """
    <div class="section-title">
        🆘 ASSISTANCE BREAKDOWN
    </div>
    """,
    unsafe_allow_html=True
)

assistance_counts = (
    disaster_data["assistance"].value_counts()
)

for assistance, count in assistance_counts.items():

    st.write(
        f"**{assistance} — {count:,}**"
    )
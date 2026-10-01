import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="IDS-ML Security Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --------------------------------------------------
# MODERN LIGHT UI
# --------------------------------------------------

st.markdown("""
<style>

/* ==================================================
   GLOBAL
================================================== */

.stApp {
    background: #F7F9FC;
}

.block-container {
    max-width: 1450px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}

/* Main text */
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
    color: #25324B;
}

/* ==================================================
   SIDEBAR
================================================== */

section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E8ECF3;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #44516A;
}

/* Navigation options */
section[data-testid="stSidebar"] label {
    border-radius: 9px;
    padding: 6px 8px;
    transition: all 0.15s ease;
}

section[data-testid="stSidebar"] label:hover {
    background-color: #F1F5FF;
}

/* ==================================================
   TITLES
================================================== */
h1 { 
    color: #17233C !important; 
    font-size: 2rem !important; 
    font-weight: 700 !important; 
    letter-spacing: -0.7px; 
} 
 
h2 { 
    color: #25324B !important; 
    font-weight: 650 !important; 
    letter-spacing: -0.3px; 
} 
 
h3 { 
    color: #34415A !important; 
    font-weight: 600 !important; 
}

/* Normal paragraph text */
p {
    color: #647089;
}

/* ==================================================
   METRIC CARDS
================================================== */

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E8ECF3;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow:
        0 1px 2px rgba(16, 24, 40, 0.02),
        0 4px 14px rgba(16, 24, 40, 0.04);
}

div[data-testid="stMetricLabel"] {
    color: #7B879D;
    font-size: 0.82rem;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #1D2A44;
    font-weight: 700;
    font-size: 1.65rem;
}

/* ==================================================
   BUTTONS
================================================== */

.stButton > button {
    border-radius: 9px;
    min-height: 42px;
    font-weight: 600;
    border: 1px solid #DDE4F0;
    background: #FFFFFF;
    color: #35445F;
    transition: all 0.15s ease;
}

.stButton > button:hover {
    border-color: #7895D5;
    color: #496BAF;
    background: #F8FAFF;
}

.stButton > button[kind="primary"] {
    background: #6688CC;
    color: #FFFFFF;
    border: none;
    box-shadow: 0 3px 10px rgba(102, 136, 204, 0.18);
}

.stButton > button[kind="primary"]:hover {
    background: #5879BA;
    color: #FFFFFF;
}

/* ==================================================
   DATA TABLES
================================================== */

div[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border: 1px solid #E6EAF1;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(20, 32, 55, 0.025);
}

/* ==================================================
   INPUTS / DROPDOWNS
================================================== */

div[data-baseweb="select"] > div {
    background: #FFFFFF;
    border-color: #DDE3EC;
    border-radius: 9px;
}

div[data-testid="stNumberInput"] input {
    background: #FFFFFF;
    border-color: #DDE3EC;
    border-radius: 9px;
}

/* ==================================================
   EXPANDERS
================================================== */

div[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E6EAF1;
    border-radius: 11px;
}

/* ==================================================
   ALERTS
================================================== */

div[data-testid="stAlert"] {
    border-radius: 10px;
    border-width: 1px;
}

/* ==================================================
   DIVIDERS
================================================== */

hr {
    border: none;
    border-top: 1px solid #E8ECF3;
    margin: 1.8rem 0;
}

/* ==================================================
   STREAMLIT CLEANUP
================================================== */

footer {
    visibility: hidden;
}
/* ==================================================
   INFORMATION CARDS
================================================== */

.info-card {
    background: #FFFFFF;
    border: 1px solid #E6EAF1;
    border-radius: 14px;
    padding: 22px;
    min-height: 175px;
    box-shadow:
        0 2px 10px rgba(20, 32, 55, 0.035);
}

.info-card-icon {
    font-size: 1.5rem;
    margin-bottom: 14px;
}

.info-card-title {
    color: #7B879D;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.info-card-value {
    color: #1D2A44;
    font-size: 1.3rem;
    font-weight: 700;
    margin-top: 5px;
    margin-bottom: 8px;
}

.info-card-text {
    color: #778399;
    font-size: 0.88rem;
    line-height: 1.5;
}
/* ==================================================
   BUTTON TEXT FIX
   ================================================== */

div.stButton > button,
div.stButton > button p,
div.stButton > button span {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

div.stButton > button:hover,
div.stButton > button:hover p,
div.stButton > button:hover span {
    color: #FFFFFF !important;
}

div.stButton > button:focus,
div.stButton > button:focus p,
div.stButton > button:focus span {
    color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "UNSW_NB15_training-set.csv.gz",
        compression="gzip"
    )
    return df

@st.cache_resource
def load_model():
    model = joblib.load("random_forest_ids_deploy.pkl")
    model_features = joblib.load("model_features.pkl")

    return model, model_features

def prepare_input_for_model(input_df, model_features):

    input_df = input_df.copy()

    # Remove fields that are not predictive model inputs
    for column in ["label", "id"]:

        if column in input_df.columns:
            input_df = input_df.drop(
                columns=[column]
            )

    categorical_columns = [
        column
        for column in ["proto", "service", "state"]
        if column in input_df.columns
    ]

    encoded = pd.get_dummies(
        input_df,
        columns=categorical_columns,
        drop_first=True
    )

    # Match the exact 191-column structure
    # used when the Random Forest was trained.
    encoded = encoded.reindex(
        columns=model_features,
        fill_value=0
    )

    encoded = encoded.astype(float)

    return encoded

    # Force the input to have exactly the same
    # 191 columns used during model training
    encoded = encoded.reindex(
        columns=model_features,
        fill_value=0
    )

    # Ensure all values are numerical
    encoded = encoded.astype(float)

    return encoded
# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🛡️ IDS-ML")

    st.caption("AI-Powered Intrusion Detection")

    st.divider()

    page = st.radio(
       "Navigation",
    [
        "Dashboard",
        "Data Explorer",
        "Model Training",
        "EDA & Visualizations",
        "Model Performance",
        "Threat Detection"
    ]
      )
    st.divider()

    st.caption("UNSW-NB15 Dataset")
    st.caption("Random Forest Classifier")


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
if page == "Dashboard":

    # --------------------------------------------------
    # HEADER
    # --------------------------------------------------

    st.title("Network Security Dashboard")

    st.markdown(
        """
        AI-powered intrusion detection using the **UNSW-NB15**
        cybersecurity dataset and a trained **Random Forest classifier**.
        """
    )

    st.divider()

    # --------------------------------------------------
    # MODEL PERFORMANCE OVERVIEW
    # --------------------------------------------------

    st.subheader("Model Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            "95.98%"
        )

    with col2:
        st.metric(
            "Precision",
            "96.37%"
        )

    with col3:
        st.metric(
            "Recall",
            "97.78%"
        )

    with col4:
        st.metric(
            "F1 Score",
            "97.07%"
        )

    st.divider()

    # --------------------------------------------------
    # SYSTEM OVERVIEW
    # --------------------------------------------------

    st.subheader("System Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-icon">🛡️</div>
                <div class="info-card-title">Detection Model</div>
                <div class="info-card-value">Random Forest</div>
                <div class="info-card-text">
                    Binary classification of normal and
                    malicious network traffic.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-icon">📊</div>
                <div class="info-card-title">Dataset</div>
                <div class="info-card-value">UNSW-NB15</div>
                <div class="info-card-text">
                    Network traffic dataset used for
                    model training and evaluation.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-icon">🔍</div>
                <div class="info-card-title">Detection Type</div>
                <div class="info-card-value">Binary IDS</div>
                <div class="info-card-text">
                    Classifies traffic as either
                    Normal or Malicious.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # --------------------------------------------------
    # DETECTION SUMMARY
    # --------------------------------------------------

    st.subheader("Detection Summary")
    col1, col2 = st.columns([1.15, 1], gap="large")

    with col1:

        detection_data = pd.DataFrame({
            "Classification": [
                "Correctly Detected",
                "Incorrectly Classified"
            ],
            "Records": [
                33658,
                1411
            ]
        })

        st.bar_chart(
            detection_data,
            x="Classification",
            y="Records"
        )

    with col2:

        st.metric(
            "Correct Predictions",
            "33,658"
        )

        st.metric(
            "Misclassified Records",
            "1,411"
        )

        st.metric(
            "Malicious Traffic Detected",
            "23,338"
        )

    st.divider()

    # --------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------

    st.subheader("System Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("✓ Dataset Loaded")

    with col2:
        st.success("✓ Model Trained")

    with col3:
        st.success("✓ Detection Ready")
# --------------------------------------------------
# DATA EXPLORER
# --------------------------------------------------

elif page == "Data Explorer":

    st.title("📂 Data Explorer")

    st.write(
        "Explore the UNSW-NB15 network traffic dataset "
        "used to train the intrusion detection model."
    )

    # Load the dataset
    df = load_data()

    st.divider()

    # Dataset statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Records",
            f"{df.shape[0]:,}"
        )

    with col2:
        st.metric(
            "Total Features",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    st.divider()

    # Dataset preview
    st.subheader("Dataset Preview")

    rows = st.slider(
        "Number of rows to display",
        min_value=5,
        max_value=100,
        value=10
    )

    st.dataframe(
        df.head(rows),
        use_container_width=True
    )

    st.divider()

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Column Names**")
        st.write(list(df.columns))

    with col2:
        st.write("**Data Types**")
        st.dataframe(
            df.dtypes.astype(str).reset_index().rename(
                columns={
                    "index": "Feature",
                    0: "Data Type"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

elif page == "Model Training":

    st.title("⚙️ Model Training")

    st.write(
        "Configuration and training information for the "
        "Random Forest intrusion detection model."
    )

    # Load dataset and trained model
    df = load_data()
    model, model_features = load_model()

    st.divider()

    # --------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------

    st.subheader("Dataset Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset",
            "UNSW-NB15"
        )

    with col2:
        st.metric(
            "Total Records",
            f"{len(df):,}"
        )

    with col3:
        st.metric(
            "Model Features",
            len(model_features)
        )

    st.divider()

    # --------------------------------------------------
    # ALGORITHM
    # --------------------------------------------------

    st.subheader("Algorithm Selection")

    st.success("✓ Random Forest Classifier — Selected")

    st.write(
        "Random Forest was selected as the classification "
        "algorithm for detecting normal and malicious "
        "network traffic."
    )

    st.divider()

    # --------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------

    st.subheader("Algorithm Configuration")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Random Forest Settings**")

        st.write("Number of Trees: **100**")
        st.write("Random State: **42**")
        st.write("Parallel Processing: **Enabled (n_jobs = -1)**")

    with col2:

        st.write("**Training Settings**")

        st.write("Training Data: **80%**")
        st.write("Testing Data: **20%**")
        st.write("Stratified Split: **Enabled**")

    st.divider()

    # --------------------------------------------------
    # TRAIN / TEST DISTRIBUTION
    # --------------------------------------------------

    st.subheader("Training / Testing Distribution")

    total_records = len(df)

    training_records = int(total_records * 0.8)
    testing_records = total_records - training_records

    split_data = pd.DataFrame({
        "Dataset Split": [
            "Training Data",
            "Testing Data"
        ],
        "Records": [
            training_records,
            testing_records
        ]
    })

    st.bar_chart(
        split_data,
        x="Dataset Split",
        y="Records"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Training Records",
            f"{training_records:,}",
            "80% of dataset"
        )

    with col2:
        st.metric(
            "Testing Records",
            f"{testing_records:,}",
            "20% of dataset"
        )

    st.divider()

    # --------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------

    st.subheader("Model Status")

    st.success(
        "✓ Random Forest model is trained and loaded successfully."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", "95.98%")

    with col2:
        st.metric("Precision", "96.37%")

    with col3:
        st.metric("Recall", "97.78%")

    with col4:
        st.metric("F1 Score", "97.07%")

# --------------------------------------------------
# EDA & VISUALIZATIONS
# --------------------------------------------------

elif page == "EDA & Visualizations":

    st.title("📊 EDA & Visualizations")

    st.write(
        "Explore patterns, attack categories, and feature "
        "distributions within the UNSW-NB15 dataset."
    )

    df = load_data()

    st.divider()

    # --------------------------------------------------
    # TRAFFIC CLASSIFICATION
    # --------------------------------------------------

    st.subheader("Traffic Classification Overview")

    label_counts = df["label"].value_counts()

    normal_count = int(label_counts.get(0, 0))
    malicious_count = int(label_counts.get(1, 0))
    total = normal_count + malicious_count

    normal_percentage = (
        normal_count / total * 100 if total else 0
    )

    malicious_percentage = (
        malicious_count / total * 100 if total else 0
    )

    col1, col2 = st.columns([1, 1.5])

    with col1:

        st.metric(
            "Normal Traffic",
            f"{normal_count:,}",
            f"{normal_percentage:.1f}% of records"
        )

        st.metric(
            "Malicious Traffic",
            f"{malicious_count:,}",
            f"{malicious_percentage:.1f}% of records"
        )

        st.metric(
            "Total Records",
            f"{total:,}"
        )

    with col2:

        traffic_chart = pd.DataFrame({
            "Traffic Type": ["Normal", "Malicious"],
            "Records": [normal_count, malicious_count]
        })

        st.bar_chart(
            traffic_chart,
            x="Traffic Type",
            y="Records"
        )

    # --------------------------------------------------
    # ATTACK CATEGORIES
    # --------------------------------------------------

    st.divider()

    st.subheader("Attack Category Distribution")

    st.caption(
        "Distribution of attack categories represented "
        "within the UNSW-NB15 dataset."
    )

    attack_counts = (
        df["attack_cat"]
        .fillna("Normal")
        .value_counts()
        .reset_index()
    )

    attack_counts.columns = [
        "Attack Category",
        "Records"
    ]

    col1, col2 = st.columns([1.7, 1])

    with col1:

        st.bar_chart(
            attack_counts,
            x="Attack Category",
            y="Records"
        )

    with col2:

        st.dataframe(
            attack_counts,
            use_container_width=True,
            hide_index=True,
            height=350
        )

    # --------------------------------------------------
    # FEATURE DISTRIBUTION EXPLORER
    # --------------------------------------------------

    st.divider()

    st.subheader("Feature Distribution Explorer")

    st.caption(
        "Select a numerical network feature to inspect "
        "its distribution across the dataset."
    )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    excluded_columns = ["id", "label"]

    numeric_columns = [
        column
        for column in numeric_columns
        if column not in excluded_columns
    ]

    selected_feature = st.selectbox(
        "Network Feature",
        numeric_columns
    )

    feature_data = df[selected_feature].dropna()

    fig, ax = plt.subplots(figsize=(8, 3.8))

    ax.hist(
        feature_data,
        bins=40,
        edgecolor="white"
    )

    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Distribution of {selected_feature}",
        fontweight="bold"
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("📈 Model Performance")

    model, model_features = load_model()

    st.write(
        "Evaluation results of the trained Random Forest "
        "intrusion detection model."
    )

    st.divider()

    # --------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------

    st.subheader("Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            "95.98%"
        )

    with col2:
        st.metric(
            "Precision",
            "96.37%"
        )

    with col3:
        st.metric(
            "Recall",
            "97.78%"
        )

    with col4:
        st.metric(
            "F1 Score",
            "97.07%"
        )

    st.divider()

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    st.subheader("Confusion Matrix")

    st.caption(
        "Breakdown of correct and incorrect classifications "
        "made by the Random Forest model."
    )

    confusion_matrix = [
        [10320, 880],
        [531, 23338]
    ]

    col1, col2 = st.columns([1.3, 1])

    with col1:

        fig, ax = plt.subplots(figsize=(4.5, 3.2))

        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt=",d",
            cmap="Blues",
            cbar=False,
            linewidths=2,
            linecolor="white",
            xticklabels=["Normal", "Malicious"],
            yticklabels=["Normal", "Malicious"],
            annot_kws={
                "size": 12,
                "weight": "bold"
            },
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")

        ax.set_title(
            "Classification Results",
            fontsize=12,
            fontweight="bold",
            pad=12
        )

        fig.tight_layout()

        st.pyplot(
          fig,
          use_container_width=False
)

        plt.close(fig)

    with col2:

        st.markdown("#### Classification Breakdown")

        st.metric(
            "True Negatives",
            "10,320",
            help="Normal traffic correctly classified as normal."
        )

        st.metric(
            "True Positives",
            "23,338",
            help="Malicious traffic correctly detected."
        )

        st.metric(
            "False Positives",
            "880",
            help="Normal traffic incorrectly classified as malicious."
        )

        st.metric(
            "False Negatives",
            "531",
            help="Malicious traffic incorrectly classified as normal."
        )

    # --------------------------------------------------
    # ERROR ANALYSIS
    # --------------------------------------------------

    st.divider()

    st.subheader("Detection Error Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "False Positive Rate",
            "7.86%",
            help=(
                "Normal traffic incorrectly classified "
                "as malicious."
            )
        )

    with col2:

        st.metric(
            "False Negative Rate",
            "2.22%",
            help=(
                "Malicious traffic incorrectly classified "
                "as normal."
            )
        )

       # --------------------------------------------------
    # VISUAL PERFORMANCE ANALYSIS
    # --------------------------------------------------

    st.divider()

    st.subheader("Visual Performance Analysis")

    chart_col1, chart_col2 = st.columns(
        2,
        gap="large"
    )

    # ==================================================
    # ROC CURVE
    # ==================================================

    with chart_col1:

        st.markdown("#### ROC Curve")

        st.caption(
            "True positive rate versus false positive rate."
        )

        roc_df = load_data()

        X_roc = roc_df.drop(
            columns=["label", "id"],
            errors="ignore"
        )

        y_roc = roc_df["label"]

        X_roc = pd.get_dummies(
            X_roc,
            drop_first=True
        )

        X_roc = X_roc.reindex(
            columns=model_features,
            fill_value=0
        )

        _, X_test_roc, _, y_test_roc = train_test_split(
            X_roc,
            y_roc,
            test_size=0.2,
            random_state=42,
            stratify=y_roc
        )

        y_probability = model.predict_proba(
            X_test_roc
        )[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test_roc,
            y_probability
        )

        auc_score = roc_auc_score(
            y_test_roc,
            y_probability
        )

        fig_roc, ax_roc = plt.subplots(
            figsize=(4.7, 3.3)
        )

        ax_roc.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"Random Forest (AUC {auc_score:.3f})"
        )

        ax_roc.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1
        )

        ax_roc.set_xlabel(
            "False Positive Rate",
            fontsize=9
        )

        ax_roc.set_ylabel(
            "True Positive Rate",
            fontsize=9
        )

        ax_roc.tick_params(
            axis="both",
            labelsize=8
        )

        ax_roc.grid(
            alpha=0.15
        )

        ax_roc.legend(
            loc="lower right",
            fontsize=8,
            frameon=False
        )

        fig_roc.tight_layout()

        st.pyplot(
            fig_roc,
            use_container_width=False
        )

        plt.close(fig_roc)

    # ==================================================
    # FEATURE IMPORTANCE
    # ==================================================

    with chart_col2:

        st.markdown("#### Feature Importance")

        st.caption(
            "Top features influencing model predictions."
        )

        feature_importance = pd.DataFrame({
            "Feature": [
                "sttl",
                "ct_state_ttl",
                "rate",
                "dload",
                "sload",
                "dmean",
                "dttl",
                "ackdat",
                "sbytes",
                "ct_srv_dst"
            ],

            "Importance": [
                0.097360,
                0.073427,
                0.070849,
                0.062089,
                0.043716,
                0.042638,
                0.039085,
                0.036202,
                0.035987,
                0.033465
            ]
        })

        feature_importance = (
            feature_importance
            .sort_values(
                "Importance",
                ascending=True
            )
        )

        fig_imp, ax_imp = plt.subplots(
            figsize=(4.7, 3.3)
        )

        ax_imp.barh(
            feature_importance["Feature"],
            feature_importance["Importance"]
        )

        ax_imp.set_xlabel(
            "Importance Score",
            fontsize=9
        )

        ax_imp.tick_params(
            axis="both",
            labelsize=8
        )

        ax_imp.grid(
            axis="x",
            alpha=0.15
        )

        ax_imp.spines["top"].set_visible(False)
        ax_imp.spines["right"].set_visible(False)

        fig_imp.tight_layout()

        st.pyplot(
            fig_imp,
            use_container_width=False
        )

        plt.close(fig_imp)


# --------------------------------------------------
# THREAT DETECTION
# --------------------------------------------------

elif page == "Threat Detection":

    st.title("🔍 Threat Detection")

    st.write(
        "Analyze network traffic using the trained "
        "Random Forest intrusion detection model."
    )

    df = load_data()
    model, model_features = load_model()

    # ==================================================
    # SINGLE RECORD DETECTION
    # ==================================================

    st.divider()

    st.subheader("Single Record Detection")

    st.caption(
        "Select a genuine UNSW-NB15 network record. "
        "The known label is hidden from the model during "
        "prediction and used afterward for verification."
    )

    record_number = st.number_input(
        "Network Record",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    selected_record = df.iloc[int(record_number)]

    col1, col2, col3 = st.columns(3)

    protocol_options = sorted(
        df["proto"].dropna().unique()
    )

    service_options = sorted(
        df["service"].dropna().unique()
    )

    state_options = sorted(
        df["state"].dropna().unique()
    )

    with col1:

        protocol = st.selectbox(
            "Protocol",
            protocol_options,
            index=protocol_options.index(
                selected_record["proto"]
            )
        )

    with col2:

        service = st.selectbox(
            "Service",
            service_options,
            index=service_options.index(
                selected_record["service"]
            )
        )

    with col3:

        state = st.selectbox(
            "State",
            state_options,
            index=state_options.index(
                selected_record["state"]
            )
        )

    st.write("**Important Network Features**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Source TTL",
            selected_record["sttl"]
        )

    with col2:

        st.metric(
            "Traffic Rate",
            f"{selected_record['rate']:.2f}"
        )

    with col3:

        st.metric(
            "Source Bytes",
            selected_record["sbytes"]
        )

    with col4:

        st.metric(
            "Destination Load",
            f"{selected_record['dload']:.2f}"
        )

    with st.expander("View Complete Network Record"):

        st.dataframe(
            selected_record.to_frame(
                name="Value"
            ),
            use_container_width=True
        )

    if st.button(
        "🔍 Analyze Selected Record",
        type="primary",
        use_container_width=True
    ):

        input_record = selected_record.drop(
            labels=["label"]
        ).copy()

        input_record["proto"] = protocol
        input_record["service"] = service
        input_record["state"] = state

        input_df = pd.DataFrame(
            [input_record]
        )

        prepared_input = prepare_input_for_model(
            input_df,
            model_features
        )

        prediction = int(
            model.predict(
                prepared_input
            )[0]
        )

        probabilities = model.predict_proba(
            prepared_input
        )[0]

        confidence = (
            probabilities[prediction] * 100
        )

        actual_label = int(
            selected_record["label"]
        )

        st.divider()

        st.subheader("Detection Result")

        if prediction == 1:

            st.error(
                "⚠️ MALICIOUS TRAFFIC DETECTED"
            )

        else:

            st.success(
                "✅ NORMAL NETWORK TRAFFIC"
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Model Prediction",
                "Malicious"
                if prediction == 1
                else "Normal"
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        with col3:

            st.metric(
                "Actual Label",
                "Malicious"
                if actual_label == 1
                else "Normal"
            )

        if prediction == actual_label:

            st.success(
                "✓ Prediction matches the known dataset label."
            )

        else:

            st.warning(
                "Prediction differs from the known dataset label."
            )

    # ==================================================
    # BATCH CSV DETECTION
    # ==================================================

    st.divider()

    st.subheader("Batch CSV Detection")

    st.caption(
        "Upload UNSW-NB15-compatible network traffic. "
        "The existing trained model analyzes every row "
        "without retraining."
    )

    uploaded_file = st.file_uploader(
        "Upload UNSW-NB15-Compatible CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(
                uploaded_file
            )

        except Exception as error:

            st.error(
                f"Could not read the CSV file: {error}"
            )

        else:

            st.write("**Uploaded Dataset Preview**")

            st.dataframe(
                batch_df.head(10),
                use_container_width=True
            )

            st.caption(
                f"{len(batch_df):,} records uploaded."
            )

            # Raw dataset columns expected by our model.
            # ID is not required for prediction.
            required_raw_features = [
                column
                for column in df.columns
                if column not in ["label", "id"]
            ]

            missing_columns = [
                column
                for column in required_raw_features
                if column not in batch_df.columns
            ]

            if missing_columns:

                st.error(
                    "This CSV is not compatible with the "
                    "trained model. Missing columns: "
                    + ", ".join(missing_columns)
                )

            else:

                batch_input = batch_df.copy()

                # Save labels for optional verification.
                uploaded_labels = None

                if "label" in batch_input.columns:

                    uploaded_labels = (
                        batch_input["label"]
                        .copy()
                        .reset_index(drop=True)
                    )

                    batch_input = batch_input.drop(
                        columns=["label"]
                    )

                # ID is metadata, not a predictive feature.
                if "id" in batch_input.columns:

                    batch_input = batch_input.drop(
                        columns=["id"]
                    )

                prepared_batch = prepare_input_for_model(
                    batch_input,
                    model_features
                )

                st.success(
                    "✓ CSV structure is compatible with "
                    "the trained model."
                )

                if st.button(
                    "Run Batch Detection",
                    type="primary",
                    use_container_width=True
                ):

                    predictions = model.predict(
                        prepared_batch
                    ).astype(int)

                    probabilities = model.predict_proba(
                        prepared_batch
                    )

                    results = batch_df.copy()

                    results["Prediction"] = [
                        "Malicious"
                        if value == 1
                        else "Normal"
                        for value in predictions
                    ]

                    results["Confidence"] = [
                        round(
                            float(
                                max(probability) * 100
                            ),
                            2
                        )
                        for probability in probabilities
                    ]

                    results[
                        "Confidence"
                    ] = (
                        results["Confidence"]
                        .astype(str)
                        + "%"
                    )

                    malicious_predictions = int(
                        (predictions == 1).sum()
                    )

                    normal_predictions = int(
                        (predictions == 0).sum()
                    )

                    st.success(
                        f"Successfully analyzed "
                        f"{len(results):,} records."
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Records Analyzed",
                            f"{len(results):,}"
                        )

                    with col2:

                        st.metric(
                            "Normal Predictions",
                            f"{normal_predictions:,}"
                        )

                    with col3:

                        st.metric(
                            "Malicious Predictions",
                            f"{malicious_predictions:,}"
                        )

                    # If uploaded CSV contains real labels,
                    # automatically verify predictions.
                    if uploaded_labels is not None:

                        actual_values = (
                            uploaded_labels
                            .astype(int)
                            .to_numpy()
                        )

                        correct = int(
                            (
                                predictions
                                == actual_values
                            ).sum()
                        )

                        batch_accuracy = (
                            correct
                            / len(predictions)
                            * 100
                        )

                        st.metric(
                            "Uploaded Sample Accuracy",
                            f"{batch_accuracy:.2f}%"
                        )

                        results[
                            "Actual Label"
                        ] = [
                            "Malicious"
                            if value == 1
                            else "Normal"
                            for value in actual_values
                        ]

                        results[
                            "Prediction Correct"
                        ] = (
                            predictions
                            == actual_values
                        )

                        st.markdown(
    """
    <h3 style="color: red !important;">
        Batch Detection Results
    </h3>
    """,
    unsafe_allow_html=True
)
                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    csv_results = (
                        results
                        .to_csv(index=False)
                        .encode("utf-8")
                    )

                    st.download_button(
                        label=(
                            "⬇️ Download Prediction Results"
                        ),
                        data=csv_results,
                        file_name=(
                            "IDS_prediction_results.csv"
                        ),
                        mime="text/csv"
                    )
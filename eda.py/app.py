import streamlit as st
import pandas as pd
import plotly.express as px
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Smart Waste Segregation",
    page_icon="♻️",
    layout="wide"
)

# --------------------------------
# TITLE
# --------------------------------

st.title("♻️ Smart Waste Segregation Dashboard")
st.write("AI-Based Waste Classification System")

# --------------------------------
# LOAD DATA
# --------------------------------

df = pd.read_csv("predictions.csv")
# --------------------------------
# LOAD AI MODEL
# --------------------------------

model = tf.keras.models.load_model(
    "waste_classifier.keras"
)

classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]
# --------------------------------
# SIDEBAR FILTERS
# --------------------------------

st.sidebar.header("🔍 Filters")

# Waste Type Filter
waste_types = sorted(
    df["Predicted Class"].dropna().unique()
)

selected_waste = st.sidebar.multiselect(
    "♻️ Select Waste Type",
    waste_types,
    default=waste_types
)
filtered_df = df[
    df["Predicted Class"].isin(selected_waste)
]
# --------------------------------
# KPI CALCULATIONS
# --------------------------------

total_images = len(filtered_df)

total_waste_types = filtered_df["Predicted Class"].nunique()

if not filtered_df.empty:

    most_detected = (
        filtered_df["Predicted Class"]
        .value_counts()
        .idxmax()
    )

    average_confidence = filtered_df["Confidence"].mean()

else:

    most_detected = "N/A"
    average_confidence = 0

# If confidence is stored as decimal
if average_confidence <= 1:
    average_confidence = average_confidence * 100

# --------------------------------
# KPI CARDS
# --------------------------------

st.markdown("""
<style>

.kpi-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e6e6e6;
    text-align: center;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
    min-height: 130px;
}

.kpi-icon {
    font-size: 28px;
    margin-bottom: 5px;
}

.kpi-title {
    font-size: 15px;
    color: #666666;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 27px;
    font-weight: bold;
    color: #222222;
}


/* --------------------------------
   REAL-TIME PREDICTION CARDS
   -------------------------------- */

.realtime-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e6e6e6;
    text-align: center;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
    min-height: 140px;
}

.realtime-icon {
    font-size: 30px;
    margin-bottom: 5px;
}

.realtime-title {
    font-size: 15px;
    color: #666666;
    margin-bottom: 10px;
}

.realtime-value {
    font-size: 25px;
    font-weight: bold;
    color: #222222;
}

</style>
""", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">📷</div>
        <div class="kpi-title">Total Images</div>
        <div class="kpi-value">{total_images}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">♻️</div>
        <div class="kpi-title">Waste Types</div>
        <div class="kpi-value">{total_waste_types}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🏆</div>
        <div class="kpi-title">Most Detected Waste</div>
        <div class="kpi-value">{most_detected.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🎯</div>
        <div class="kpi-title">Average Confidence</div>
        <div class="kpi-value">{average_confidence:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------
# WASTE DISTRIBUTION
# --------------------------------

waste_counts = (
    filtered_df["Predicted Class"]
    .value_counts()
    .reset_index()
)

waste_counts.columns = [
    "Waste Type",
    "Count"
]

# --------------------------------
# BAR CHART
# --------------------------------

fig = px.bar(
    waste_counts,
    x="Waste Type",
    y="Count",
    title="Number of Predictions by Waste Type",
    text="Count"
)

fig.update_traces(
    textposition="outside"
)

fig.update_layout(
    showlegend=False,
    margin=dict(l=20, r=20, t=50, b=20)
)

# --------------------------------
# DONUT CHART
# --------------------------------

fig_pie = px.pie(
    waste_counts,
    names="Waste Type",
    values="Count",
    hole=0.5
)

fig_pie.update_traces(
    textposition="inside",
    textinfo="percent+label"
)

fig_pie.update_layout(
    showlegend=True,
    margin=dict(l=20, r=20, t=50, b=20)
)

# --------------------------------
# DISPLAY CHARTS SIDE-BY-SIDE
# --------------------------------

st.subheader("📊 Waste Distribution")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )
# --------------------------------
# PREDICTION DATA
# --------------------------------

st.subheader("📋 Prediction Data")

st.dataframe(
    filtered_df,
    use_container_width=True
)
# --------------------------------
# AI WASTE CLASSIFICATION
# --------------------------------

st.subheader("🤖 Classify New Waste Image")

uploaded_files = st.file_uploader(
    "Upload waste images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:

    # --------------------------------
    # REAL-TIME UPLOADED IMAGE ANALYSIS
    # --------------------------------

    results = []

    for uploaded_file in uploaded_files:

        # Open uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        # Resize image
        image_resized = image.resize((224, 224))

        # Convert image to NumPy array
        img_array = np.array(image_resized)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(
            img_array,
            verbose=0
        )

        # Find predicted class
        predicted_index = np.argmax(prediction)

        predicted_class = classes[predicted_index]

        # Calculate confidence
        confidence = float(
            np.max(prediction) * 100
        )

        # Confidence status
        if confidence >= 80:
            status = "High"

        elif confidence >= 60:
            status = "Medium"

        else:
            status = "Low"

        # Store result
        results.append({
            "Image": uploaded_file.name,
            "Predicted Waste": predicted_class,
            "Confidence": confidence,
            "Status": status
        })

    # Convert results into DataFrame
    realtime_df = pd.DataFrame(results)

    # --------------------------------
    # OVERALL ANALYTICAL DASHBOARD
    # --------------------------------

    st.subheader(
        "📊 Overall Real-Time Analysis"
    )

    # KPI calculations

    total_uploaded = len(realtime_df)

    most_detected_uploaded = (
        realtime_df["Predicted Waste"]
        .value_counts()
        .idxmax()
    )

    average_uploaded_confidence = (
        realtime_df["Confidence"].mean()
    )

    high_confidence = (
        realtime_df["Status"] == "High"
    ).sum()

    medium_confidence = (
        realtime_df["Status"] == "Medium"
    ).sum()

    low_confidence = (
        realtime_df["Status"] == "Low"
    ).sum()

    # --------------------------------
    # ANALYTICAL CARDS
    # --------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="realtime-card">
            <div class="realtime-icon">📷</div>
            <div class="realtime-title">
                Total Uploaded
            </div>
            <div class="realtime-value">
                {total_uploaded}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="realtime-card">
            <div class="realtime-icon">♻️</div>
            <div class="realtime-title">
                Most Detected Waste
            </div>
            <div class="realtime-value">
                {most_detected_uploaded.upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="realtime-card">
            <div class="realtime-icon">🎯</div>
            <div class="realtime-title">
                Average Confidence
            </div>
            <div class="realtime-value">
                {average_uploaded_confidence:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="realtime-card">
            <div class="realtime-icon">🟢</div>
            <div class="realtime-title">
                High Confidence
            </div>
            <div class="realtime-value">
                {high_confidence}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------
    # CONFIDENCE STATUS ANALYSIS
    # --------------------------------

    st.subheader(
        "🎯 Confidence Status Analysis"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="realtime-card">
            <div class="realtime-icon">🟢</div>
            <div class="realtime-title">
                High Confidence
            </div>
            <div class="realtime-value">
                {high_confidence}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="realtime-card">
            <div class="realtime-icon">🟡</div>
            <div class="realtime-title">
                Medium Confidence
            </div>
            <div class="realtime-value">
                {medium_confidence}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="realtime-card">
            <div class="realtime-icon">🔴</div>
            <div class="realtime-title">
                Low Confidence
            </div>
            <div class="realtime-value">
                {low_confidence}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # --------------------------------
    # CONFIDENCE STATUS CHART
    # --------------------------------

    confidence_data = pd.DataFrame({
        "Status": ["High", "Medium", "Low"],
        "Count": [
            high_confidence,
            medium_confidence,
            low_confidence
        ]
    })

    fig_confidence = px.pie(
        confidence_data,
        names="Status",
        values="Count",
        hole=0.5,
        title="Confidence Status Distribution"
    )

    fig_confidence.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    fig_confidence.update_layout(
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(
        fig_confidence,
        use_container_width=True
    )

    # --------------------------------
    # DISPLAY UPLOADED IMAGES
    # --------------------------------

    st.subheader(
        "🖼️ Uploaded Image Predictions"
    )

    st.dataframe(
        realtime_df,
        use_container_width=True
    )


            
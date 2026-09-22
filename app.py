import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 18px;
    margin-bottom: 35px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    text-align: center;
    margin-top: 20px;
}

.result-title {
    font-size: 18px;
    color: #6b7280;
}

.result-value {
    font-size: 32px;
    font-weight: 700;
    margin-top: 8px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #eef6ff;
    border-left: 5px solid #2563eb;
    margin-top: 20px;
}

.warning-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #fff8e6;
    border-left: 5px solid #f59e0b;
    margin-top: 25px;
}

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

from huggingface_hub import hf_hub_download

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="saiSushanth777/brain-tumor-svm",
        filename="brain_tumor_hog_svm.pkl"
    )

    return joblib.load(model_path)


model = load_model()


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_image(image):

    # PIL → NumPy
    img = np.array(image)

    # RGB → Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    img = clahe.apply(img)

    # Resize
    img = cv2.resize(
        img,
        (100, 100),
        interpolation=cv2.INTER_LINEAR
    )

    # HOG
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    # Convert to 2D
    features = features.reshape(1, -1)

    return features


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🧠 Brain Tumor MRI Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered MRI image classification using HOG + Tuned SVM'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write("**Model:** Tuned SVM")
    st.write("**Features:** HOG")
    st.write("**Preprocessing:** CLAHE + Grayscale")
    st.write("**Image Size:** 100 × 100")

    st.divider()

    st.subheader("📋 Classes")

    st.write("🟠 Glioma")
    st.write("🔵 Meningioma")
    st.write("🟢 No Tumor")
    st.write("🟣 Pituitary")

    st.divider()

    st.caption(
        "Model trained using the Brain Tumor MRI dataset."
    )


# =========================================================
# FILE UPLOAD
# =========================================================

st.subheader("📤 Upload MRI Image")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# IMAGE DISPLAY
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:

        st.markdown("### 🖼️ Uploaded MRI")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.markdown("### 🔍 Image Information")

        st.write(
            f"**File name:** {uploaded_file.name}"
        )

        st.write(
            f"**Image size:** {image.size[0]} × {image.size[1]}"
        )

        st.write(
            f"**Image mode:** {image.mode}"
        )

        st.info(
            "The image will be converted to grayscale, "
            "enhanced using CLAHE, and converted into HOG features."
        )


    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    st.markdown("---")

    predict_button = st.button(
        "🔎 Analyze MRI",
        use_container_width=True,
        type="primary"
    )


    if predict_button:

        with st.spinner("Analyzing MRI..."):

            # Preprocess
            features = preprocess_image(image)

            # Prediction
            prediction = model.predict(features)[0]

            # Class names
            class_names = {
                0: "Glioma",
                1: "Meningioma",
                2: "No Tumor",
                3: "Pituitary"
            }

            result = class_names[prediction]


        # =================================================
        # RESULT
        # =================================================

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-title">Model Prediction</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-value">{result}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # =================================================
        # DECISION SCORES
        # =================================================

        st.markdown("### 📊 Model Decision Scores")

        scores = model.decision_function(features)

        score_values = scores[0]

        score_data = {
            "Class": [
                "Glioma",
                "Meningioma",
                "No Tumor",
                "Pituitary"
            ],
            "Score": score_values
        }

        st.dataframe(
            score_data,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # INTERPRETATION
        # =================================================

        if result == "No Tumor":

            st.success(
                "The model classified this image as No Tumor."
            )

        else:

            st.warning(
                f"The model classified this image as {result}."
            )


        # =================================================
        # DISCLAIMER
        # =================================================

        st.markdown(
            '<div class="warning-box">'
            '⚠️ <b>Important:</b> This application is an '
            'educational/research machine-learning project. '
            'The prediction should not be considered a medical '
            'diagnosis or a substitute for professional medical advice.'
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# INITIAL MESSAGE
# =========================================================

else:

    st.markdown(
        """
        <div class="info-box">
        <b>👆 Get started:</b> Upload an MRI image above to begin
        classification.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
    Brain Tumor MRI Classification • HOG + Tuned SVM
    </div>
    """,
    unsafe_allow_html=True
)
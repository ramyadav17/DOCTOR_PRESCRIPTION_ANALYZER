import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract

# Path to tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Page config
st.set_page_config(page_title="Prescription Analyzer", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Upload and analyze prescription images")

show_processed = st.sidebar.checkbox("Show Processed Image", True)

# Title
st.markdown("<h1 style='text-align: center;'>💊 Prescription Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a prescription image and extract text easily</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Upload Prescription Image", type=["jpg", "png", "jpeg"])

# Image processing
def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

# OCR
def extract_text(image):
    return pytesseract.image_to_string(image)

# Layout columns
col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)

    processed = process_image(img_array)

    if show_processed:
        with col2:
            st.subheader("🧪 Processed Image")
            st.image(processed, use_column_width=True)

    # Analyze button
    if st.button("🔍 Analyze Prescription"):
        with st.spinner("Analyzing..."):
            result = extract_text(processed)

        st.success("✅ Analysis Complete")

        st.subheader("📄 Extracted Text")
        st.text_area("Result", result, height=250)

        # Download option
        st.download_button(
            label="📥 Download Text",
            data=result,
            file_name="prescription_text.txt",
            mime="text/plain"
        )

else:
    st.info("Please upload an image to begin.")
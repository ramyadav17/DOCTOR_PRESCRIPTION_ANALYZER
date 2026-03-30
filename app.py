import streamlit as st
from PIL import Image
import tempfile
import pandas as pd

from backend import (
    extract_text_from_image,
    extract_structured_data,
    classify_lines,
)

st.set_page_config(page_title="Prescription Analyzer", layout="wide")

st.title("Prescription Analyzer")
st.markdown("Upload a prescription image and extract structured medical data.")

# Sidebar
st.sidebar.header("⚙️ Settings")
show_steps = st.sidebar.checkbox("Show Pipeline Steps", True)

# Upload
uploaded_file = st.file_uploader("Upload Prescription Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    # ===== PIPELINE =====
    with col2:
        st.subheader("Processing Pipeline")

        # OCR
        text = extract_text_from_image(image_path)

        if show_steps:
            st.markdown("### OCR Output")
            st.code(text)

        # Structured Extraction
        structured_data = extract_structured_data(text)

        if show_steps:
            st.markdown("### Structured Extraction")
            st.json(structured_data)

        # Classification
        instruction = classify_lines(text)

        if show_steps:
            st.markdown("### Instruction Classification")
            st.write(instruction)

    # ===== FINAL OUTPUT =====
    st.divider()
    st.subheader("Prescription Summary")

    # -------- MEDICINES --------
    st.markdown("### Medicines")

    if structured_data:
        medicines = list(set([row["Medicine"] for row in structured_data]))

        cols = st.columns(len(medicines))
        for i, med in enumerate(medicines):
            cols[i].markdown(
                f"""
                <div style="
                    padding:10px;
                    border-radius:10px;
                    background-color:#1f2937;
                    text-align:center;
                    font-weight:bold;
                    color:#22c55e;">
                    {med.upper()}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("No medicines detected")

    # -------- TABLE --------
    st.markdown("### Prescription Details")

    if structured_data:
        df = pd.DataFrame(structured_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No structured data extracted")

    # -------- INSTRUCTION --------
    st.markdown("### Instruction Type")
    if instruction:
        df_instr = pd.DataFrame(instruction)

        st.dataframe(
            df_instr,
            use_container_width=True
        )
    else:
        st.info("No instructions detected")
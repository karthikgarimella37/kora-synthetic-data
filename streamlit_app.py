import os
import tempfile
import pickle
from io import StringIO, BytesIO
from typing import Optional, Tuple

import faiss
import pandas as pd
import streamlit as st

from pdf_parser import pdf_parser
from generate_embedding import (
    load_embedding_model as load_embed_model,
    generate_embedding,
    generate_faiss_index,
)
from response_generator import SyntheticDataGenerator
from synthetic_data_scaling import scale_csv_text


def extract_csv_text(raw_text: str) -> str:
    """Best-effort extraction of CSV content from an LLM response."""
    text = raw_text.strip()
    # Try fenced blocks first
    if "```" in text:
        parts = text.split("```")
        # Prefer block labeled csv
        for i in range(len(parts) - 1):
            if parts[i].lower().strip().endswith("csv"):
                return parts[i + 1].strip()
        # Otherwise take the first fenced content
        if len(parts) >= 2:
            return parts[1].strip()
    return text


def split_explanation_and_csv(raw_text: str) -> Tuple[str, str]:
    """Split the LLM response into (explanation_text, csv_text).

    Prefers a fenced block labeled ```csv for CSV; if absent, falls back to the first
    fenced block. Everything outside the chosen fenced block is treated as explanation.
    """
    text = raw_text or ""
    lines = text.splitlines()

    # First, try to find a fenced block explicitly labeled as csv
    csv_start = None
    csv_end = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if csv_start is None and s.lower() == "```csv":
            csv_start = i + 1
            continue
        if csv_start is not None and s == "```":
            csv_end = i
            break

    # If not found, fallback to the first fenced block of any type
    if csv_start is None or csv_end is None:
        start = None
        end = None
        for i, ln in enumerate(lines):
            if ln.strip().startswith("```"):
                start = i + 1
                break
        if start is not None:
            for j in range(start, len(lines)):
                if lines[j].strip() == "```":
                    end = j
                    break
        if start is not None and end is not None:
            csv_start, csv_end = start, end

    if csv_start is not None and csv_end is not None and 0 <= csv_start <= csv_end <= len(lines):
        csv_lines = lines[csv_start:csv_end]
        explanation_lines = lines[:csv_start - 1] + lines[csv_end + 1:]
        explanation_text = "\n".join(explanation_lines).strip()
        csv_text = "\n".join(csv_lines).strip()
        return explanation_text, csv_text

    # If no fences at all, assume the whole text is CSV (worst-case) and no explanation
    return "", text.strip()


st.set_page_config(page_title="Healthcare Paper → Synthetic Data", layout="wide")
st.title("Kora Synthetic Healthcare Data Generation")

with st.sidebar:
    # st.header("Settings & API Keys")
    # llama_key = st.text_input("LLAMA_CLOUD_API_KEY", type="password")
    # openrouter_key = st.text_input("OPENROUTER_API_KEY", type="password")
    # dbtwin_key = st.text_input("DBTWIN_API_KEY (optional for scaling)", type="password")

    model_name = st.text_input("LLM Model (OpenRouter)", value="google/gemini-flash-1.5")
    k_chunks = st.number_input("Top K chunks for context", min_value=3, max_value=30, value=10, step=1)
    scale_rows = st.number_input("Rows for scaled dataset (DBTwin)", min_value=100, max_value=200000, value=1000, step=100)

    st.markdown("Provide keys here or set them in your environment.")

uploaded_pdf = st.file_uploader("Upload a research paper PDF", type=["pdf"])

col_left, col_right = st.columns(2)

# Initialize session state
if "df_sample" not in st.session_state:
    st.session_state.df_sample = None
if "sample_csv_bytes" not in st.session_state:
    st.session_state.sample_csv_bytes = None
if "explanation_text" not in st.session_state:
    st.session_state.explanation_text = None

sample_csv_bytes: Optional[bytes] = None
scaled_csv_bytes: Optional[bytes] = None

if uploaded_pdf:
    process = st.button("Process PDF → Generate Sample CSV → (Optional) Scale")
    if process:
        # Apply provided keys to environment for downstream libraries
        if llama_key:
            os.environ["LLAMA_CLOUD_API_KEY"] = llama_key
        if openrouter_key:
            os.environ["OPENROUTER_API_KEY"] = openrouter_key
        if dbtwin_key:
            os.environ["DBTWIN_API_KEY"] = dbtwin_key

        with st.spinner("Parsing PDF into chunks..."):
            pdf_content = pdf_parser(uploaded_pdf)
            st.success(f"Parsed {len(pdf_content)} chunks from PDF")

        with st.spinner("Loading embedding model and generating embeddings..."):
            embed_model = load_embed_model("all-MiniLM-L6-v2")
            embeddings = generate_embedding(pdf_content, embed_model)
            index = generate_faiss_index(embeddings)
            st.success("Embeddings created and FAISS index built")

        # Persist index and chunks so the response generator can load them
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "faiss_index.index")
            chunks_path = os.path.join(tmpdir, "text_chunks.pkl")
            faiss.write_index(index, index_path)
            with open(chunks_path, "wb") as f:
                pickle.dump(pdf_content, f)

            with st.spinner("Generating sample CSV using LLM and retrieved context..."):
                try:
                    generator = SyntheticDataGenerator(
                        openai_api_key=openrouter_key or os.getenv("OPENROUTER_API_KEY"),
                        model_name=model_name,
                        index_path=index_path,
                        text_chunks_path=chunks_path,
                        max_context_length=8000,
                    )
                    result = generator.generate_synthetic_data(k=int(k_chunks))
                    raw_response = result.get("response", "")
                    explanation_text, csv_text = split_explanation_and_csv(raw_response)
                    # Try parsing to validate
                    csv_text_clean = "\n".join([ln for ln in csv_text.splitlines() if ln.strip() not in ('csv', 'CSV', '```', '```csv', '```CSV')])
                    df_sample = pd.read_csv(StringIO(csv_text_clean), comment='`')
                    df_sample.rename(columns=lambda c: str(c).strip(), inplace=True)
                    if 'csv' in df_sample.columns or 'CSV' in df_sample.columns:
                        df_sample = df_sample.drop(columns=[c for c in ('csv', 'CSV') if c in df_sample.columns])
                    sample_csv_bytes = df_sample.to_csv(index=False).encode()
                    
                    # Store in session state
                    st.session_state.df_sample = df_sample
                    st.session_state.sample_csv_bytes = sample_csv_bytes
                    st.session_state.explanation_text = explanation_text
                    
                    st.success("Sample CSV generated from LLM")
                except Exception as e:
                    st.error(f"Failed to generate or parse sample CSV: {e}")
                    st.stop()

# Display the results outside the process button block
if st.session_state.df_sample is not None:
    with col_left:
        st.subheader("Sample CSV Preview")
        st.dataframe(st.session_state.df_sample.head(20))
        st.download_button(
            "Download sample CSV",
            data=st.session_state.sample_csv_bytes,
            file_name="synthetic_data.csv",
            mime="text/csv",
        )

    with col_right:
        st.subheader("Feature Summary from Paper")
        if st.session_state.explanation_text:
            st.markdown(st.session_state.explanation_text)
        else:
            st.caption("No explanation provided by the model.")

        st.subheader("Scale via DBTwin (optional)")
        if not (dbtwin_key or os.getenv("DBTWIN_API_KEY")):
            st.info("Provide DBTWIN_API_KEY in sidebar to enable scaling.")
        else:
            if st.button("Scale sample CSV with DBTwin"):
                with st.spinner("Scaling dataset via DBTwin API..."):
                    try:
                        scaled_df, scaled_bytes, headers = scale_csv_text(
                            st.session_state.df_sample.to_csv(index=False),
                            dbtwin_key or os.getenv("DBTWIN_API_KEY"),
                            rows=int(scale_rows),
                            algo="flagship",
                        )
                        scaled_csv_bytes = scaled_bytes
                        st.success("Scaled dataset generated")
                        st.dataframe(scaled_df.head(20))

                        dist_err = headers.get("distribution-similarity-error") if headers else None
                        assoc_sim = headers.get("association-similarity") if headers else None
                        if dist_err or assoc_sim:
                            st.caption(f"distribution-similarity-error: {dist_err} | association-similarity: {assoc_sim}")

                        st.download_button(
                            "Download scaled CSV",
                            data=scaled_csv_bytes,
                            file_name="synthetic_generated_data.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Scaling failed: {e}")

else:
    st.info("Upload a PDF to begin.")


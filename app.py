import streamlit as st
from transformers import pipeline
import torch

# App configuration
st.set_page_config(
    page_title="Real or AI",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            padding: 0.75rem;
            border-radius: 8px;
            font-weight: 600;
            background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
            border: none;
            color: white;
        }
        .result {
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            background: #f8fafc;
        }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("Real or AI")
st.write("Paste text to detect if it was written by a human or AI")

# Text input
text = st.text_area("Enter text to analyze (minimum 50 characters)", height=200)

# Analyze button
if st.button("Analyze Text"):
    if len(text) < 50:
        st.warning("Please enter at least 50 characters for better accuracy")
    else:
        with st.spinner("Analyzing..."):
            # Load model
            @st.cache_resource
            def load_model():
                return pipeline(
                    "text-classification",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if torch.cuda.is_available() else -1
                )
            
            model = load_model()
            
            # Get prediction
            result = model(text)[0]
            label = result['label']
            score = result['score'] * 100
            
            # Display result
            st.markdown(f"""
                <div class="result">
                    <h3>Analysis Result</h3>
                    <p>This text is <strong>{"Human-written" if label == "POSITIVE" else "AI-generated"}</strong> with {score:.1f}% confidence</p>
                    <div style="height: 10px; background: #e2e8f0; border-radius: 5px; margin: 1rem 0;">
                        <div style="width: {score}%; height: 100%; background: #6366f1; border-radius: 5px;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.9rem;">
        <p>Built with ❤️ using Streamlit and Hugging Face</p>
    </div>
""", unsafe_allow_html=True)

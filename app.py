import streamlit as st
import torch
from transformers import pipeline

# App configuration
st.set_page_config(
    page_title="Real or AI",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        :root {
            --primary: #6366f1;
            --background: #0f172a;
            --text: #f8fafc;
        }
        body { color: var(--text); background: var(--background); }
        .stTextArea textarea { 
            background-color: rgba(255,255,255,0.05) !important; 
            color: var(--text) !important; 
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load a simple text classification model"""
    try:
        # Using a small, efficient model
        return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title("🔍 Real or AI")
    st.markdown("### Simple AI Text Detector")
    
    # Load model
    classifier = load_model()
    if classifier is None:
        st.error("Failed to load the AI model. Please try refreshing the page.")
        return
    
    # Main UI
    text = st.text_area("Enter text to analyze (minimum 20 characters):", height=200)
    
    if st.button("Analyze Text", type="primary"):
        if not text.strip():
            st.warning("Please enter some text to analyze")
        elif len(text.strip()) < 20:
            st.warning("Please enter at least 20 characters")
        else:
            with st.spinner("Analyzing..."):
                try:
                    # Simple analysis
                    result = classifier(text[:512])[0]  # Limit to first 512 tokens
                    score = result['score']
                    label = result['label']
                    
                    # Map sentiment to human/AI (this is a simplification)
                    is_human = label == "POSITIVE"
                    human_prob = score * 100 if is_human else (100 - (score * 100))
                    ai_prob = 100 - human_prob
                    
                    # Display results
                    st.subheader("Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Human", f"{human_prob:.1f}%")
                    with col2:
                        st.metric("AI", f"{ai_prob:.1f}%")
                    
                    st.info("Note: This is a simplified demo using sentiment analysis as a proxy for human/AI detection.")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()

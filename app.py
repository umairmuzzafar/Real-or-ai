import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# App configuration
st.set_page_config(
    page_title="Real or AI",
    page_icon="🤖",
    layout="wide"
)

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"Using device: {device}")

@st.cache_resource
def load_model():
    try:
        model_name = "facebook/bart-large-mnli"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# UI
st.title("Real or AI Detector")
text = st.text_area("Enter text to analyze", height=200)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text to analyze")
    else:
        model, tokenizer = load_model()
        if model is None or tokenizer is None:
            st.error("Failed to load the model. Please check the logs.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    
                    human_score = probs[0][1].item() * 100
                    ai_score = probs[0][0].item() * 100
                    
                    st.success(f"Human: {human_score:.1f}% | AI: {ai_score:.1f}%")
                    st.progress(int(human_score))
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time

# App configuration
APP_NAME = "Real or AI"
APP_ICON = "🔍"
MODEL_NAME = "facebook/bart-large-mnli"  # Using a more reliable model

# Modern 2025 design with custom CSS
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --background: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
            --success: #10b981;
            --warning: #f59e0b;
        }
        
        body {
            color: var(--text);
            background-color: var(--background);
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }
        
        .stTextArea textarea {
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: var(--text) !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 16px;
            font-size: 16px;
            min-height: 200px;
        }
        
        .stButton>button {
            background: linear-gradient(90deg, var(--primary) 0%, var(--primary-dark) 100%);
            border: none;
            color: white;
            padding: 14px 28px;
            text-align: center;
            font-size: 16px;
            margin: 10px 0;
            border-radius: 12px;
            width: 100%;
            transition: all 0.3s ease;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the model and tokenizer with error handling"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            device_map="auto" if torch.cuda.is_available() else None
        )
        if not torch.cuda.is_available():
            model = model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, ""

def main():
    st.title(f"{APP_ICON} {APP_NAME}")
    st.markdown("### Detect if text was written by a human or AI")
    
    # Load model
    model, tokenizer, device = load_model()
    
    if model is None or tokenizer is None:
        st.error("Failed to load the AI model. Please try refreshing the page.")
        return
    
    # Main UI
    text = st.text_area("Enter text to analyze:", height=200)
    
    if st.button("Analyze Text", type="primary"):
        if not text.strip():
            st.warning("Please enter some text to analyze")
        else:
            with st.spinner("Analyzing..."):
                try:
                    # Simple analysis using the model
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Get probabilities
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                    human_prob = probs[0][0].item() * 100
                    ai_prob = probs[0][1].item() * 100
                    
                    # Display results
                    st.subheader("Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Human", f"{human_prob:.1f}%")
                    with col2:
                        st.metric("AI", f"{ai_prob:.1f}%")
                    
                    if human_prob > ai_prob:
                        st.success("This text is likely human-written")
                    else:
                        st.warning("This text is likely AI-generated")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()

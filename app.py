import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

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
        .confidence-bar {
            height: 10px;
            background: #e2e8f0;
            border-radius: 5px;
            margin: 1rem 0;
            overflow: hidden;
        }
        .confidence-level {
            height: 100%;
            border-radius: 5px;
            background: #6366f1;
        }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("Real or AI")
st.write("Paste text to detect if it was written by a human or AI")

# Text input
text = st.text_area("Enter text to analyze (minimum 100 characters for better accuracy)", height=200)

@st.cache_resource
def load_model():
    # Using a model specifically trained to detect AI-generated text
    model_name = "roberta-base-openai-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

def analyze_text(text):
    model, tokenizer = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Convert to probabilities
    probabilities = predictions[0].tolist()
    return {
        "human": probabilities[1],  # Human-written
        "ai": probabilities[0]      # AI-generated
    }

# Analyze button
if st.button("Analyze Text"):
    if len(text) < 100:
        st.warning("Please enter at least 100 characters for better accuracy")
    else:
        with st.spinner("Analyzing text (this may take a moment)..."):
            try:
                result = analyze_text(text)
                human_score = result["human"] * 100
                ai_score = result["ai"] * 100
                is_human = human_score > ai_score
                
                st.markdown(f"""
                    <div class="result">
                        <h3>Analysis Result</h3>
                        <p>This text is <strong>{"Human-written" if is_human else "AI-generated"}</strong></p>
                        
                        <p>Confidence:</p>
                        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                            <span>Human-written: {human_score:.1f}%</span>
                            <span>AI-generated: {ai_score:.1f}%</span>
                        </div>
                        
                        <div class="confidence-bar">
                            <div class="confidence-level" style="width: {human_score}%;"></div>
                        </div>
                        
                        <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                            <small>0%</small>
                            <small>100%</small>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Add some analysis tips
                if ai_score > 70:
                    st.info("""
                    **AI Detection Tips:**
                    - The text shows strong patterns typical of AI generation
                    - Look for overly formal or generic language
                    - Check for lack of personal experiences or specific details
                    """)
                elif human_score > 70:
                    st.info("""
                    **Human-written Indicators:**
                    - The text shows natural variations in style
                    - Contains personal or specific details
                    - May include informal language or idioms
                    """)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please try again with a different text or check your internet connection.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.9rem;">
        <p>Built with ❤️ using Streamlit and Hugging Face</p>
    </div>
""", unsafe_allow_html=True)

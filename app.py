import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time
from typing import Tuple

# App configuration
APP_NAME = "Real or AI"
APP_ICON = "🔍"
MODEL_NAME = "facebook/bart-large-mnli"

# Modern 2025 design with custom CSS
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(f"""
    <style>
        :root {{
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --background: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
            --success: #10b981;
            --warning: #f59e0b;
        }}
        
        body {{
            color: var(--text);
            background-color: var(--background);
            font-family: 'Inter', sans-serif;
        }}
        
        .stApp {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }}
        
        .stTextArea textarea {{
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: var(--text) !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 16px;
            font-size: 16px;
            min-height: 200px;
        }}
        
        .stButton>button {{
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
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
        }}
        
        .result-card {{
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 24px;
            margin: 20px 0;
            border-left: 5px solid var(--primary);
        }}
        
        .confidence-meter {{
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            margin: 12px 0;
            overflow: hidden;
        }}
        
        .confidence-level {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }}
        
        .human {{
            background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        }}
        
        .ai {{
            background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%);
        }}
        
        .header {{
            background: linear-gradient(90deg, var(--primary) 0%, var(--primary-dark) 100%);
            padding: 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #ffffff 0%, #e2e8f0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .header p {{
            margin: 0.5rem 0 0;
            opacity: 0.9;
            font-size: 1.1rem;
        }}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=f"Loading {MODEL_NAME}...")
def load_model():
    """Load the model and tokenizer"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if not torch.cuda.is_available():
            model = model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, ""

def predict(text: str, model, tokenizer, device: str) -> Tuple[float, float]:
    """Predict if the text is AI or human-generated using zero-shot classification"""
    try:
        # Using zero-shot classification
        candidate_labels = ["human-written", "AI-generated"]
        inputs = tokenizer([text], candidate_labels, return_tensors="pt", 
                          padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.softmax(dim=1)
            
        # Get probabilities
        human_prob = logits[0][0].item()
        ai_prob = logits[0][1].item()
        
        return human_prob, ai_prob
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0.5, 0.5  # Return neutral probabilities on error

def main():
    # Header section
    st.markdown(f"""
        <div class="header">
            <h1>{APP_NAME} {APP_ICON}</h1>
            <p>Detect if a text was written by a human or generated by AI</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, tokenizer, device = load_model()
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📝 Enter Text to Analyze")
        text = st.text_area(
            "Paste any text below to check if it was written by a human or AI:",
            placeholder="Type or paste text here...",
            label_visibility="collapsed"
        )
        
        analyze_btn = st.button("Analyze Text", type="primary", disabled=not model)
    
    with col2:
        st.markdown("### 🔍 Analysis Results")
        
        if 'result' not in st.session_state:
            st.session_state.result = None
        
        if analyze_btn and text.strip():
            with st.spinner("Analyzing text..."):
                try:
                    start_time = time.time()
                    human_prob, ai_prob = predict(text, model, tokenizer, device)
                    st.session_state.result = {
                        'human_prob': human_prob,
                        'ai_prob': ai_prob,
                        'text': text[:200] + "..." if len(text) > 200 else text
                    }
                    st.success(f"Analysis completed in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.session_state.result = None
        
        if st.session_state.result:
            result = st.session_state.result
            is_human = result['human_prob'] > result['ai_prob']
            confidence = max(result['human_prob'], result['ai_prob']) * 100
            
            st.markdown(f"""
                <div class="result-card">
                    <h3 style="margin-top: 0; color: {'#10b981' if is_human else '#f59e0b'}">
                        {'Human-written' if is_human else 'AI-generated'}
                    </h3>
                    <p>Confidence:</p>
                    <div class="confidence-meter">
                        <div class="confidence-level {'human' if is_human else 'ai'}" 
                             style="width: {confidence}%"></div>
                    </div>
                    <p style="text-align: right; margin: 0; opacity: 0.8;">
                        {confidence:.1f}% confident
                    </p>
                </div>
                
                <div style="margin-top: 24px; padding: 16px; background: rgba(255,255,255,0.05); border-radius: 12px;">
                    <h4 style="margin-top: 0;">📊 Detailed Analysis</h4>
                    <div style="margin-bottom: 16px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span>Human-written</span>
                            <span style="font-weight: 600; color: #10b981;">{result['human_prob']*100:.1f}%</span>
                        </div>
                        <div class="confidence-meter">
                            <div class="confidence-level human" style="width: {result['human_prob']*100}%"></div>
                        </div>
                    </div>
                    <div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span>AI-generated</span>
                            <span style="font-weight: 600; color: #f59e0b;">{result['ai_prob']*100:.1f}%</span>
                        </div>
                        <div class="confidence-meter">
                            <div class="confidence-level ai" style="width: {result['ai_prob']*100}%"></div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # Add some space and footer
    st.markdown("""
        <div style="margin-top: 60px; padding: 20px; text-align: center; opacity: 0.7; font-size: 0.9rem;">
            <hr style="border: 0; height: 1px; background: rgba(255, 255, 255, 0.1); margin: 20px 0;">
            <p>🔍 Real or AI - An open-source AI text detection tool</p>
            <p>Built with ❤️ using <a href="https://huggingface.co/" target="_blank">Hugging Face</a> and <a href="https://streamlit.io/" target="_blank">Streamlit</a></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

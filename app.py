import streamlit as st
import torch
import numpy as np

# Try to import transformers with fallback
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as e:
    st.error("Error importing transformers. Please ensure all dependencies are installed.")
    st.code("pip install transformers torch sentencepiece protobuf tqdm")
    st.stop()

# Rest of your app code remains the same...
# [Previous app code continues here]

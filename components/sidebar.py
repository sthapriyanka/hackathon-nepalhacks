import streamlit as st

def configure_sidebar():
    """
    Configure the sidebar with all settings.
    
    Returns:
        dict: Dictionary containing all configuration settings
    """
    st.sidebar.markdown('<h2 class="sub-header">Configuration</h2>', unsafe_allow_html=True)
    
    # Model selection (placeholder for future model options)
    model_option = "LLama-2-7B"  # Default model
    
    # QA Generation settings
    st.sidebar.markdown('<h3>QA Generation Settings</h3>', unsafe_allow_html=True)
    # min_confidence = st.sidebar.slider("Minimum Confidence Score", 0.0, 1.0, 0.7)
    min_confidence=0.7
    max_qa_pairs = st.sidebar.slider("Maximum QA Pairs", 5, 50, 20)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)
    advanced_mode=False
    # advanced_mode = st.sidebar.checkbox("Advanced Mode", False)
    
    # Advanced settings
    # if advanced_mode:
    #     top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.95)
    #     ocr_resolution = st.sidebar.slider("OCR Resolution (DPI)", 150, 600, 300)
    # else:
    # temperature = 0.2
    top_p = 0.95
    ocr_resolution = 300
    
    # Return all settings as a dictionary
    return {
        "model": model_option,
        "min_confidence": min_confidence,
        "max_qa_pairs": max_qa_pairs,
        "advanced_mode": advanced_mode,
        "temperature": temperature,
        "top_p": top_p,
        "ocr_resolution": ocr_resolution
    }
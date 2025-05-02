import streamlit as st
import os
import atexit
from components.sidebar import configure_sidebar
from components.display import show_instructions, show_example
from core.processor import DocumentProcessor

# Set page title and configuration
st.set_page_config(
    page_title="Document-to-QnA Pair Converter",
    page_icon="❓",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2563EB;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #F59E0B;
    }
    .qa-question {
        font-weight: bold;
        color: #1E3A8A;
    }
    .qa-answer {
        color: #1F2937;
    }
    .qa-confidence {
        font-size: 0.9rem;
        color: #6B7280;
    }
    .qa-confidence {
        font-size: 0.8rem;
        color: #6B7280;
    }
    .qa-flag {
        color: #DC2626;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Document-to-QnA Pair Converter</h1>', unsafe_allow_html=True)
st.markdown('<div class="info-box">Convert security policies, compliance documents, and evidence files into structured question-answer pairs.</div>', unsafe_allow_html=True)

def cleanup():
    """Clean up temporary files on session end."""
    from utils.file_utils import cleanup_temp_files
    cleanup_temp_files()

def main():
    """Main application entry point."""
    # Configure sidebar and get settings
    settings = configure_sidebar()
    
    # Initialize document processor with settings
    processor = DocumentProcessor(settings)
    
    # Process document
    processor.process_document()
    
    # Show instructions and examples
    with st.expander("How to Use This Tool", expanded=False):
        show_instructions()
    
    with st.expander("Example Input & Output", expanded=False):
        show_example()

if __name__ == "__main__":
    main()
    # Register cleanup function
    atexit.register(cleanup)
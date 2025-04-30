import streamlit as st
import os
from utils.text_processing import load_spacy_model
from utils.file_utils import save_uploaded_file
from utils.ui_utils import display_qa_pairs, show_download_options
from components.display import show_file_details
from core.parser import extract_text, clean_text, segment_text
from core.qna import generate_qa_pairs

class DocumentProcessor:
    """Main document processing pipeline class."""
    
    def __init__(self, settings):
        """
        Initialize the document processor.
        
        Args:
            settings (dict): Configuration settings from sidebar
        """
        self.settings = settings
        self.nlp = load_spacy_model()
        
    def process_document(self):
        """Process an uploaded document and generate QA pairs."""
        # File uploader
        st.markdown('<h2 class="sub-header">Upload Document</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a PDF, DOCX, or image file",
            type=["pdf", "docx", "doc", "jpg", "jpeg", "png", "txt"]
        )
        
        if uploaded_file is None:
            return
        
        # Display file info
        show_file_details(uploaded_file)
        
        # Save uploaded file temporarily
        temp_path = save_uploaded_file(uploaded_file)
        
        try:
            with st.spinner("Processing document..."):
                # Extract text based on file type
                file_type = uploaded_file.name.split(".")[-1].lower()
                extracted_text = extract_text(temp_path, file_type)
                
                # Clean and segment text
                cleaned_text = clean_text(extracted_text)
                text_segments = segment_text(self.nlp, cleaned_text)
                
                # Show text extraction results in an expandable section
                with st.expander("View Extracted Text", expanded=False):
                    st.text_area("Extracted text", extracted_text, height=300)
                
                # Process segments to generate QA pairs
                all_qa_pairs = self._generate_qa_pairs(text_segments)
                
                # Filter by confidence score
                filtered_qa_pairs = [qa for qa in all_qa_pairs if qa["confidence"] >= self.settings["min_confidence"]]
                
                # Limit to max_qa_pairs
                filtered_qa_pairs = filtered_qa_pairs[:self.settings["max_qa_pairs"]]
                
                # Show QA pairs
                st.markdown('<h2 class="sub-header">Generated QA Pairs</h2>', unsafe_allow_html=True)
                st.write(f"Generated {len(filtered_qa_pairs)} QA pairs (filtered from {len(all_qa_pairs)} total)")
                
                # Display QA pairs with filtering and sorting options
                display_df = display_qa_pairs(filtered_qa_pairs)
                
                # Show download options
                show_download_options(filtered_qa_pairs)
                
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
        
        finally:
            # Clean up the temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _generate_qa_pairs(self, text_segments):
        """
        Generate QA pairs from text segments with progress bar.
        
        Args:
            text_segments (list): List of text segments
            
        Returns:
            list: List of QA pair dictionaries
        """
        all_qa_pairs = []
        
        with st.spinner("Generating QA pairs... This may take a few minutes."):
            # Progress bar
            progress_bar = st.progress(0)
            
            # Process each segment
            for i, segment in enumerate(text_segments):
                segment_qa_pairs = generate_qa_pairs(
                    segment, 
                    max_pairs=max(1, int(self.settings["max_qa_pairs"]/len(text_segments))),
                    temperature=self.settings["temperature"],
                    top_p=self.settings["top_p"]
                )
                all_qa_pairs.extend(segment_qa_pairs)
                
                # Update progress
                progress_bar.progress((i + 1) / len(text_segments))
        
        return all_qa_pairs
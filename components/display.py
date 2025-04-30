import streamlit as st

def show_instructions():
    """Display instructions on how to use the tool."""
    st.markdown("""
    ### Instructions
    1. **Upload Document**: Select a PDF, DOCX, or image file containing policy or compliance text.
    2. **Configure Settings**: Adjust model parameters and filtering options in the sidebar.
    3. **View Results**: Explore generated QA pairs and their confidence scores.
    4. **Export**: Download results in CSV or JSON format for further use.
    
    ### Tips for Best Results
    - For complex documents, consider splitting them into smaller parts before uploading.
    - Adjust the confidence threshold to filter out low-quality pairs.
    - Check flagged items manually, as they may require human verification.
    - For images, higher resolution may improve OCR quality but increases processing time.
    """)

def show_example():
    """Display example input and output."""
    st.markdown("""
    ### Example Input Text
    ```
    The Acceptable Use Policy (AUP) outlines the requirements for proper use of company 
    information and technology resources. All employees must sign the AUP during onboarding 
    and annually thereafter. The policy covers electronic transmission security including 
    email, web, and file transfer services protocols. All users are responsible for ensuring 
    the security of their assigned devices including mobile phones and laptops.
    ```
    
    ### Example QA Pairs Generated
    1. **Q**: Is there an acceptable use policy for company information and technology resources?  
       **A**: The Acceptable Use Policy (AUP) outlines the requirements for proper use of company information and technology resources.  
       **Confidence**: 0.95
       
    2. **Q**: Are employees required to sign the Acceptable Use Policy?  
       **A**: All employees must sign the AUP during onboarding and annually thereafter.  
       **Confidence**: 0.92
       
    3. **Q**: Does the policy cover electronic transmission security requirements?  
       **A**: The policy covers electronic transmission security including email, web, and file transfer services protocols.  
       **Confidence**: 0.89
       
    4. **Q**: Who is responsible for ensuring device security?  
       **A**: All users are responsible for ensuring the security of their assigned devices including mobile phones and laptops.  
       **Confidence**: 0.85
    """)

def show_file_details(file):
    """
    Display file details.
    
    Args:
        file: The uploaded file
    """
    file_details = {"Filename": file.name, "File size": f"{file.size / 1024:.2f} KB"}
    st.write(file_details)
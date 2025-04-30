import os
import tempfile
import base64

def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file to a temporary location and return the path.
    
    Args:
        uploaded_file: The uploaded file from Streamlit
        
    Returns:
        str: Path to the saved temporary file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return temp_file.name

def get_download_link(df, filename, text):
    """
    Generate a download link for a dataframe.
    
    Args:
        df: Pandas DataFrame to download
        filename: Name of the file to download
        text: Text to display for the download link
        
    Returns:
        str: HTML link for downloading the DataFrame as CSV
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def get_json_download_link(data, filename, text):
    """
    Generate a download link for JSON data.
    
    Args:
        data: Data to convert to JSON
        filename: Name of the file to download
        text: Text to display for the download link
        
    Returns:
        str: HTML link for downloading the data as JSON
    """
    import json
    json_str = json.dumps(data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">{text}</a>'
    return href

def cleanup_temp_files():
    """Clean up temporary files created during the session."""
    temp_dir = tempfile.gettempdir()
    for file in os.listdir(temp_dir):
        if file.startswith('tmp'):
            try:
                os.unlink(os.path.join(temp_dir, file))
            except:
                pass
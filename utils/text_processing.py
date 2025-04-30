import spacy

def load_spacy_model():
    """
    Load the spaCy model for NER and text processing.
    
    Returns:
        spacy.Language: Loaded spaCy model
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        # Download if not installed
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    return nlp

def chunk_text(text, max_chunk_size=1000):
    """
    Split text into manageable chunks for processing.
    
    Args:
        text (str): The text to chunk
        max_chunk_size (int): Maximum size of each chunk in characters
        
    Returns:
        list: List of text chunks
    """
    # Simple chunking by character count
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        
        # Try to find a sentence ending or paragraph break
        if end < len(text):
            # Look for sentence ending or paragraph break
            for i in range(end, max(start, end - 200), -1):
                if text[i] in ['.', '!', '?', '\n'] and (i+1 >= len(text) or text[i+1].isspace()):
                    end = i + 1
                    break
        
        chunks.append(text[start:end])
        start = end
        
    return chunks
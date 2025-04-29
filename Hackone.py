import pdfplumber
import subprocess
import ollama

# --- Step 1: Extract text from PDF ---
def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text.strip()

# --- Step 2: Generate QnA using Ollama ---
def generate_qna_with_ollama(text, model="mistral", num_qna=5):
    prompt = f"Generate {num_qna} clear question-answer pairs from the following policy text:\n\n{text}\n\nOutput in json format."
    
    response = ollama.chat(model='mistral', messages=[
        {"role": "user", "content": prompt}
    ])
    # process = subprocess.Popen(
    #     ["ollama", "run", model],
    #     stdin=subprocess.PIPE,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE
    # )
    print(response['message']['content'])

# --- Step 3: Main workflow ---
if __name__ == "__main__":
    pdf_file = "Vulnerability Management Policy.docx.pdf"  # Replace with your actual file name
    print("📄 Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_file)

    if not extracted_text:
        print("❌ No text extracted. Is this a scanned PDF?")
    else:
        print("🧠 Generating QnA using Ollama...")
        result = generate_qna_with_ollama(extracted_text)
        print("\n--- Generated Q&A Pairs ---\n")
        print(result)

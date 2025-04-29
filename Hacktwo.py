import pdfplumber
import json
import ollama
import pandas as pd

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
    prompt = f"Generate {num_qna} clear question-answer pairs from the following policy text:\n\n{text}\n\nOutput in JSON format with keys: 'question' and 'answer' so that i can extract it to file json."

    response = ollama.chat(model=model, messages=[
        {"role": "user", "content": prompt}
    ])

    raw_output = response['message']['content']

    try:
        qna_list = json.loads(raw_output)
        return qna_list
    except json.JSONDecodeError:
        print("⚠️ Error: Output is not valid JSON.")
        print("Raw Output:\n", raw_output)
        return None

def save_qna_to_excel(qna_list, filename="qna_output.xlsx"):
    df = pd.DataFrame(qna_list)
    df.to_excel(filename, index=False)
    print(f"📁 Q&A saved to Excel file: {filename}")

# --- Step 3: Main workflow ---
if __name__ == "__main__":
    pdf_file = "Vulnerability Management Policy.docx.pdf"
    print("📄 Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_file)

    if not extracted_text:
        print("❌ No text extracted. Is this a scanned PDF?")
    else:
        print("🧠 Generating QnA using Ollama...")
        qna_data = generate_qna_with_ollama(extracted_text)

        if qna_data:
            print("\n✅ Parsed Q&A JSON Output:\n")
            for item in qna_data:
                print(f"Q: {item['question']}\nA: {item['answer']}\n")

            save_qna_to_excel(qna_data)


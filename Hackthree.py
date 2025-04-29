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

# --- Step 2: Generate QnA with Question Type Classification ---
def generate_qna_with_ollama(text, model="mistral"):
    prompt = f"""
    Analyze the following text and extract ALL questions. Classify each question into:
    1. *Knowledge-based* (factual, definitions, procedures)
    2. *Research-methodology* (study design, data collection, analysis)
    3. *Other* (if unclear or irrelevant).

    Return a JSON with the structure:
    {{
      "knowledge_based": [{{"question": "...", "answer": "..."}}],
      "methodology": [{{"question": "...", "answer": "..."}}],
      "other": [{{"question": "...", "answer": "..."}}]
    }}

    Text:
    {text}
    """

    response = ollama.chat(model=model, messages=[
        {"role": "user", "content": prompt}
    ])

    raw_output = response['message']['content']
    
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        print("⚠ JSON Error. Raw output:\n", raw_output)
        return None

# --- Step 3: Save to Excel with Question Types ---
def save_qna_to_excel(qna_data, filename="classified_qna.xlsx"):
    # Flatten the categorized Q&A into a single DataFrame
    rows = []
    for category in qna_data:
        for item in qna_data[category]:
            rows.append({"Type": category, "Question": item["question"], "Answer": item["answer"]})
    
    df = pd.DataFrame(rows)
    df.to_excel(filename, index=False)
    print(f"📁 Saved classified Q&A to {filename}")

# --- Main Workflow ---
if __name__ == "__main__":
    pdf_file = "Vulnerability Management Policy.docx.pdf"
    print("📄 Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_file)

    if not extracted_text:
        print("❌ No text extracted. Is this a scanned PDF?")
    else:
        print("🧠 Generating classified Q&A...")
        qna_data = generate_qna_with_ollama(extracted_text)

        if qna_data:
            print("\n✅ Classified Questions:")
            for category in qna_data:
                print(f"\n--- {category.upper()} ---")
                for item in qna_data[category]:
                    print(f"Q: {item['question']}\nA: {item['answer']}\n")

            save_qna_to_excel(qna_data)
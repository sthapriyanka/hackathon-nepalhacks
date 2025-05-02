import pdfplumber
from PIL import Image
import pytesseract
import docx2txt
import re
import os
import layoutparser as lp
import numpy as np

def extract_text_from_pdf(pdf_file, image_output_dir="output_tables"):
    """
    Extracts structured text from a PDF file including:
    - Plain text
    - OCR from detected tables in embedded images
    
    Args:
        pdf_file (str): Path to the PDF file.
        image_output_dir (str): Directory to store extracted table images.

    Returns:
        str: Consolidated extracted content (text + tables).
    """
    os.makedirs(image_output_dir, exist_ok=True)

    # Load LayoutParser model
    model = lp.models.Detectron2LayoutModel(
        config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]
    )

    full_extracted_text = ""

    with pdfplumber.open(pdf_file) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            page_text = f"\n--- Page {page_index} ---\n"

            # Extract embedded images and process for tables
            for img_index, img_dict in enumerate(page.images):
                bbox = (img_dict["x0"], img_dict["top"], img_dict["x1"], img_dict["bottom"])
                cropped_image = page.crop(bbox).to_image(resolution=300)
                image_path = os.path.join(image_output_dir, f"page_{page_index}_img_{img_index+1}.png")
                cropped_image.save(image_path)

                image_pil = Image.open(image_path).convert("RGB")
                image_np = np.array(image_pil)
                layout = model.detect(image_np)

                tables = [b for b in layout if b.type == "Table"]
                if not tables:
                    tables = [lp.TextBlock(lp.Rectangle(0, 0, image_pil.width, image_pil.height), type="Table")]

                for t_index, table in enumerate(tables):
                    cropped_table = table.crop_image(image_np)
                    cropped_table_pil = Image.fromarray(cropped_table)
                    ocr_text = pytesseract.image_to_string(cropped_table_pil, config="--psm 6")
                    page_text += f"\nTable {img_index+1}.{t_index+1} OCR Raw:\n{ocr_text.strip()}\n"

            # Extract plain text
            text = page.extract_text()
            if text:
                page_text += f"\n--- Page {page_index} Plain Text ---\n{text}"

            full_extracted_text += page_text

    print(full_extracted_text);
    return full_extracted_text


def extract_text_from_docx(docx_file):
    """Extract text from DOCX file."""
    return docx2txt.process(docx_file)

def extract_text_from_image(image_file, ocr_resolution=300):
    """Extract text from image using OCR."""
    image = Image.open(image_file)
    # Increase resolution for better OCR results
    text = pytesseract.image_to_string(image, config=f'--dpi {ocr_resolution}')
    return text

def extract_text(file_path, file_type):
    """Extract text based on file type."""
    if file_type == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_type in ["docx", "doc"]:
        return extract_text_from_docx(file_path)
    elif file_type in ["jpg", "jpeg", "png"]:
        return extract_text_from_image(file_path)
    elif file_type == "txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "Unsupported file format"

def clean_text(text):
    """Clean extracted text for better processing."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might interfere with model
    text = re.sub(r'[^\w\s.,;:?!()\[\]{}\-\'""`]', '', text)
    return text.strip()

def segment_text(nlp, text, max_length=1000):
    """
    Segment long text into manageable, semantically meaningful chunks using spaCy.
    """
    doc = nlp(text[:1000000])  # Limit to 1 million characters for safety

    segments = []
    current_segment = ""

    for sent in doc.sents:
        sentence = sent.text.strip()
        if len(current_segment) + len(sentence) + 1 > max_length:
            segments.append(current_segment.strip())
            current_segment = sentence
        else:
            current_segment += " " + sentence if current_segment else sentence

    if current_segment:
        segments.append(current_segment.strip())

    return segments
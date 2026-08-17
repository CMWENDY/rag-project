import os
import json
from pypdf import PdfReader
from bs4 import BeautifulSoup

RAW_DIR = "docs/raw"
PROCESSED_DIR = "docs/processed"

def load_text_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def load_pdf_file(filepath):
    reader = PdfReader(filepath)
    pages_text = []
    for page_number, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        pages_text.append({"page": page_number + 1, 
                            "text": page_text})
    return pages_text

def load_html_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text


def load_document(filepath):
    extension = os.path.splitext(filepath)[1].lower()

    if extension in [".txt", ".md"]:
        text = load_text_file(filepath)
        return [{"page": None, "text": text}]
    
    elif extension == ".pdf":
        return load_pdf_file(filepath)
    
    elif extension in [".html",".htm"]:
        text = load_html_file(filepath)
        return [{"page": None, "text": text}]
    
    else:
        return None

# Walking the folder + processing every file 
def load_all_documents():
    documents = []

    for root, dirs, files in os.walk(RAW_DIR):
        for filename in files:
            filepath = os.path.join(root, filename)
            pages = load_document(filepath)

            if pages is None:
                print(f"Skipping unsupported file: {filepath}")
                continue
            
            for page_entry in pages:
                documents.append({
                    "source": filename,
                    "page": page_entry["page"],
                    "text": page_entry["text"].strip()
                })
    return documents

# Saving the processed output
def save_processed_documents(documents):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, "documents.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)
    
    print(f"Saved {len(documents)} document entries to {output_path}")


if __name__ == "__main__":
    docs = load_all_documents()
    save_processed_documents(docs)
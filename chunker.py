import os
import json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import numpy as np
from openai import OpenAI

load_dotenv()
client= OpenAI()


INPUT_FILE = "docs/processed/documents.json"
OUTPUT_FILE = "docs/processed/chunks.json"

def fixed_size_chunks(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def structure_aware_chunks(text, chunk_size=800, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in sentences if s.strip()]

def embed_sentences(sentences, batch_size=100):
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings

def cosine_similarity(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_chunks(text, similarity_threshold=0.75):
    sentences = split_into_sentences(text)
    if len(sentences) < 2:
        return [text]
    
    embeddings = embed_sentences(sentences)
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim < similarity_threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    
    chunks.append(" ".join(current_chunk))
    return chunks

def entity_from_source(source):
    # Country / legal-entity specific benefits files
    if "inc-benefits-us" in source:        return "us"
    if "canada-corp" in source:            return "canada"
    if "ltd-benefits-uk" in source:        return "uk"
    if "bv-benefits-belgium" in source:    return "belgium"
    if "bv-benefits-finland" in source:    return "finland"
    if "bv-benefits-netherlands" in source: return "netherlands"
    if "france-sas" in source:             return "france"
    if "korea-ltd" in source:              return "korea"
    if "gitlab-ireland-ltd" in source:     return "ireland"
    if "pty-benefits-australia" in source: return "australia"
    if "pty-benefits-new-zealand" in source: return "new-zealand"
    if "singapore-pte-ltd" in source:      return "singapore"
    if "gitlab-gk" in source:              return "japan"
    if "papaya-global-benefits" in source: return "italy"

    # Country-specific NON-benefits docs (policies / legal notices). These have
    # generic-looking names but country-specific *content* (verified by reading
    # the docs, not the filename). France policy files already match "france-sas"
    # above, so they need no rule here.
    if "policies-inc-usa" in source:            return "us"
    if "leave-of-absence-us" in source:         return "us"
    if "labor-and-employment-notices" in source: return "us"
    if "policies-ireland-ltd" in source:        return "ireland"
    if "policies-india" in source:              return "india"

    # Everything else (general handbook pages, multi-country/EOR pages like
    # remote-com, financial-wellness, the benefits index) isn't tied to one
    # country.
    return "global"
    
def load_documents():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def build_chunks():
    documents = load_documents()
    all_chunks = []
    chunk_id = 0

    for doc in documents:
        text = doc["text"].strip()
        if not text:
            continue
        
        strategies = {
            "fixed": fixed_size_chunks(text),
            "structure": structure_aware_chunks(text),
            "semantic": semantic_chunks(text),
        }

        for strategy_name, chunk_texts in strategies.items():
            for chunk_text in chunk_texts:
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "source": doc["source"],
                    "page": doc["page"],
                    "strategy": strategy_name,
                    "entity": entity_from_source(doc["source"]),
                    "char_count": len(chunk_text),
                    "text": chunk_text
                })
                chunk_id += 1

    return all_chunks

def save_chunks(chunks):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to {OUTPUT_FILE}")

if __name__ == "__main__":
    chunks = build_chunks()
    save_chunks(chunks)
import os, glob, json
from chunker import (fixed_size_chunks, structure_aware_chunks,
                     semantic_chunks, entity_from_source)
import store

SAMPLE_DIR = "sample_docs"

def load_sample_documents():
    docs = []
    for path in sorted(glob.glob(f"{SAMPLE_DIR}/*.md")):
        with open(path, encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "page": 1, "text": f.read()})
    return docs

def build_chunks(documents):
    all_chunks, chunk_id = [], 0
    for doc in documents:
        text = doc["text"].strip()
        if not text:
            continue
        strategies = {
            "fixed": fixed_size_chunks(text),
            "structure": structure_aware_chunks(text),
            "semantic": semantic_chunks(text),
        }
        for strategy_name, texts in strategies.items():
            for t in texts:
                all_chunks.append({
                    "chunk_id": chunk_id, "source": doc["source"], "page": doc["page"],
                    "strategy": strategy_name, "entity": entity_from_source(doc["source"]),
                    "char_count": len(t), "text": t,
                })
                chunk_id += 1
    return all_chunks

if __name__ == "__main__":
    documents = load_sample_documents()
    chunks = build_chunks(documents)
    os.makedirs("docs/processed", exist_ok=True)
    with open("docs/processed/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    store.build_chroma_collection(chunks)   # embeds chunks and sends them to the chroma service
    store.build_bm25_index(chunks)          # writes the BM25 keyword index to docs/processed
    print(f"Seeded {len(chunks)} chunks from {len(documents)} sample documents.")
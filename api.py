import os
import json
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import retriever
import store
from pipeline import answer_question
from chunker import (fixed_size_chunks, structure_aware_chunks,
                     semantic_chunks, entity_from_source)

CHUNKS_FILE = "docs/processed/chunks.json"

# ---- auth ----
DEMO_KEY = os.getenv("DEMO_KEY")
def require_api_key(x_api_key: str | None = Header(default=None)):
    if not DEMO_KEY:
        return
    if x_api_key != DEMO_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

app = FastAPI(title="RAG Q&A API", version="1.0.0")

# ---- rate limiting ----
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------- POST /v1/ask ----------
class AskRequest(BaseModel):
    question: str
    entity: Optional[str] = None
    strategy: str = "structure"
    top_k: int = 15

@app.post("/v1/ask")
@limiter.limit("5/minute")
def ask(request: Request, req: AskRequest):
    result = answer_question(req.question, top_k=req.top_k,
                             entity=req.entity, strategy=req.strategy)
    chunks = result.pop("chunks", [])
    result["retrieved"] = [
        {"rank": i, "source": c["source"], "preview": c["text"][:200]}
        for i, c in enumerate(chunks, start=1)
    ]
    return result

# ---------- POST /v1/retrieve --------

class RetrieveRequest(BaseModel):
    question: str
    entity: Optional[str] = None
    strategy: str = "structure"
    top_k: int = 10
    mode: str = "hybrid"          # "hybrid" or "dense"

@app.post("/v1/retrieve")
def retrieve(req: RetrieveRequest):
    if req.mode == "dense":
        ids = retriever.dense_search(req.question, top_k=req.top_k,
                                     entity=req.entity, strategy=req.strategy)
        chunks = [retriever.chunk_lookup[i] for i in ids]
    else:
        chunks = retriever.hybrid_search(req.question, top_k=req.top_k,
                                         entity=req.entity, strategy=req.strategy)
    return {
        "mode": req.mode,
        "retrieved": [
            {"rank": i, "source": c["source"], "preview": c["text"][:200]}
            for i, c in enumerate(chunks, start=1)
        ],
    }

# ---------- GET /v1/documents ----------
@app.get("/v1/documents")
def list_documents():
    agg = {}
    for c in retriever.chunks:
        s = c["source"]
        if s not in agg:
            agg[s] = {"source": s, "entity": c["entity"],
                      "strategies": set(), "chunk_count": 0}
        agg[s]["strategies"].add(c["strategy"])
        agg[s]["chunk_count"] += 1
    documents = [{**d, "strategies": sorted(d["strategies"])} for d in agg.values()]
    return {"count": len(documents),
            "documents": sorted(documents, key=lambda d: d["source"])}

# ---------- POST /v1/ingest ----------
class IngestRequest(BaseModel):
    source: str
    text: str
    page: int = 1

# ingest requires the key; ask stays open (but rate-limited)
@app.post("/v1/ingest", dependencies=[Depends(require_api_key)])
def ingest(req: IngestRequest):
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    next_id = max((c["chunk_id"] for c in all_chunks), default=-1) + 1

    entity = entity_from_source(req.source)
    strategies = {
        "fixed": fixed_size_chunks(req.text),
        "structure": structure_aware_chunks(req.text),
        "semantic": semantic_chunks(req.text),
    }
    new_chunks = []
    for strategy_name, texts in strategies.items():
        for t in texts:
            new_chunks.append({
                "chunk_id": next_id, "source": req.source, "page": req.page,
                "strategy": strategy_name, "entity": entity,
                "char_count": len(t), "text": t,
            })
            next_id += 1

    if not new_chunks:
        raise HTTPException(status_code=400, detail="Document produced no chunks.")

    all_chunks.extend(new_chunks)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    store.build_chroma_collection(new_chunks)
    store.build_bm25_index(all_chunks)
    retriever.refresh()

    return {"source": req.source, "entity": entity, "chunks_added": len(new_chunks)}

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import ClapModel, ClapProcessor
import librosa
import numpy as np
import chromadb
import tempfile
import os
import uuid

app = FastAPI()

# Allow calls from your Node backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CLAP model once on startup
print("Loading CLAP model...")
model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
print("CLAP model ready.")

# ChromaDB client
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(name="music")

def get_audio_embedding(file_path: str) -> list:
    """Convert an audio file to a CLAP embedding."""
    audio, sr = librosa.load(file_path, sr=48000, mono=True)
    inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
    output = model.get_audio_features(**inputs)
    # output is a tensor directly in newer transformers versions
    if hasattr(output, 'pooler_output'):
        tensor = output.pooler_output
    else:
        tensor = output
    return tensor.detach().numpy()[0].tolist()


def get_text_embedding(text: str) -> list:
    """Convert a text description to a CLAP embedding."""
    inputs = processor(text=text, return_tensors="pt", padding=True)
    output = model.get_text_features(**inputs)
    if hasattr(output, 'pooler_output'):
        tensor = output.pooler_output
    else:
        tensor = output
    return tensor.detach().numpy()[0].tolist()

@app.post("/index")
async def index_audio(file: UploadFile = File(...)):
    """Accept an audio file, embed it, store in ChromaDB."""
    # Save upload to a temp file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        embedding = get_audio_embedding(tmp_path)
        doc_id = str(uuid.uuid4())

        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[{"filename": file.filename}]
        )

        return {"status": "indexed", "id": doc_id, "filename": file.filename}
    finally:
        os.unlink(tmp_path)  # Clean up temp file


@app.post("/search")
async def search(query: dict):
    """Accept a text query, return closest matching audio files."""
    text = query.get("text", "")
    n_results = query.get("n_results", 5)

    embedding = get_text_embedding(text)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )

    # Format results
    matches = []
    for i, doc_id in enumerate(results["ids"][0]):
        matches.append({
            "id": doc_id,
            "filename": results["metadatas"][0][i]["filename"],
            "distance": results["distances"][0][i]
        })

    return {"query": text, "results": matches}


@app.get("/health")
async def health():
    return {"status": "ok"}
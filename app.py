import os
import json
import tempfile
import string
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from whistress import WhiStressInferenceClient
from utils import (
    build_phone2id_no_stress, 
    add_phone_features, 
    pos_map_dict, 
    nlp_model
)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
whistress_client = None
phone_dict = None
device = "cuda" if torch.cuda.is_available() else "cpu"

METADATA_PATH = "/datas/store162/annhung/202608_wsd_ssd_POC/exp/wordstress/best/metadata.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global whistress_client, phone_dict
    
    print(f"Loading WhiStress model on {device}...")
    with open(METADATA_PATH, "r") as fn:
        metadata = json.load(fn)

    whistress_client = WhiStressInferenceClient(device=device, metadata=metadata)
    phone_dict = build_phone2id_no_stress()
    print("WhiStress loaded successfully!")
    
    yield
    whistress_client = None

app = FastAPI(
    title="WhiStress API",
    description="API for Word-level and Phone-level Speech Stress Analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
class PhoneDetail(BaseModel):
    phone: str
    stress: int

class WordDetail(BaseModel):
    word: str
    pos: str
    sentence_stress: int
    phones: List[PhoneDetail]

class StressResponse(BaseModel):
    transcription: str
    words: List[WordDetail]

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def prepare_features_for_transcription(transcription: str):
    """Generates phone_ids and token_pos_ids required by WhiStress model."""
    clean_text = transcription.translate(str.maketrans('', '', string.punctuation))
    words = clean_text.strip().split()
    
    # 1. Phonemes
    _, phone_ids_list, _ = add_phone_features(clean_text, phone_dict)
    phone_ids_tensor = torch.tensor([phone_ids_list], dtype=torch.long, device=device)

    # 2. POS Tags
    vp_feats = nlp_model.vocab_profile_feats(words)
    pos_feats = vp_feats["pos_list"]
    pos_ids_list = [pos_map_dict.get(p, 0) for p in pos_feats]
    token_pos_tensor = torch.tensor([pos_ids_list], dtype=torch.long, device=device)

    return phone_ids_tensor, token_pos_tensor, words, pos_feats


def align_phones_to_words(words: List[str], pos_tags: List[str], phone_stresses: List[int]):
    """Aligns predicted phone stress levels back to individual words."""
    word_details = []
    phone_ptr = 0

    for idx, word in enumerate(words):
        word_phones, _, _ = add_phone_features(word, phone_dict)
        
        phone_objects = []
        for phn in word_phones:
            stress_val = 0
            if phone_ptr < len(phone_stresses):
                stress_val = int(phone_stresses[phone_ptr])
            phone_objects.append(PhoneDetail(phone=phn, stress=stress_val))
            phone_ptr += 1

        word_details.append(
            WordDetail(
                word=word,
                pos=pos_tags[idx] if idx < len(pos_tags) else "X",
                sentence_stress=0,
                phones=phone_objects
            )
        )

    return word_details

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "device": device, "service": "WhiStress Analysis API"}


@app.post("/api/predict", response_model=StressResponse)
async def predict_stress(
    file: UploadFile = File(...),
    transcription: Optional[str] = Form(None)
):
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
        raise HTTPException(status_code=400, detail="Invalid audio file format.")

    file_suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        waveform, sample_rate = torchaudio.load(tmp_path)
        audio_dict = {
            "array": waveform.squeeze(0).numpy(),
            "sampling_rate": sample_rate
        }

        # Handle features if transcription is provided
        phone_ids = None
        token_pos_ids = None
        words = []
        pos_tags = []

        if transcription and transcription.strip():
            phone_ids, token_pos_ids, words, pos_tags = prepare_features_for_transcription(transcription)

        # Run Prediction
        scored, phone_stress_preds = whistress_client.predict(
            audio=audio_dict,
            transcription=transcription if (transcription and transcription.strip()) else None,
            return_pairs=True,
            phone_ids=phone_ids,
            token_pos_ids=token_pos_ids
        )

        pred_words, word_stresses = zip(*scored)
        final_transcript = " ".join(pred_words)

        if not words:
            words = list(pred_words)
            vp_feats = nlp_model.vocab_profile_feats(words)
            pos_tags = vp_feats["pos_list"]

        # Parse Phone Predictions
        phone_stresses_list = []
        if phone_stress_preds is not None:
            phone_stresses_list = phone_stress_preds.squeeze(0).cpu().tolist()

        # Build Response Structures
        word_structures = align_phones_to_words(words, pos_tags, phone_stresses_list)

        # Attach word-level sentence stress
        for i, word_obj in enumerate(word_structures):
            if i < len(word_stresses):
                word_obj.sentence_stress = int(word_stresses[i])

        return StressResponse(
            transcription=final_transcript,
            words=word_structures
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
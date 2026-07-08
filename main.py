from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import shutil
import os
import uuid

app = FastAPI()
client = OpenAI()


@app.get("/health")
def health():
    return {
        "status": "backend working",
        "message": "Scribed backend is alive"
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with open(temp_filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe-diarize",
                file=audio_file,
                response_format="diarized_json",
                chunking_strategy="auto"
            )

        segments = []

        for segment in transcript.segments:
            segments.append({
                "speaker": segment.speaker,
                "text": segment.text,
                "start": segment.start,
                "end": segment.end
            })

        return {
            "status": "transcription complete",
            "filename": file.filename,
            "segments": segments
        }

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


class Segment(BaseModel):
    speaker: str
    text: str


class AnalyzeRequest(BaseModel):
    segments: List[Segment]


@app.post("/analyze")
def analyze_transcript(request: AnalyzeRequest):
    return {
        "status": "analysis complete",
        "summary": "The conversation shows stress, reassurance, and emotional support.",
        "positive": 60,
        "neutral": 30,
        "negative": 10,
        "anger": 25,
        "happiness": 70,
        "core": "TRANSCENDENT",
        "language_mix": {
            "english": 55,
            "arabic": 30,
            "mixed": 15
        },
        "speaker_results": [
            {
                "speaker": "ENTITY 1",
                "tone": "anxious but motivated"
            },
            {
                "speaker": "ENTITY 2",
                "tone": "calm and supportive"
            },
            {
                "speaker": "ENTITY 3",
                "tone": "reassuring"
            }
        ]
    }

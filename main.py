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
    
class LanguageMix(BaseModel):
    english: int
    arabic: int
    mixed: int


class SpeakerResult(BaseModel):
    speaker: str
    tone: str


class SentimentReport(BaseModel):
    status: str
    summary: str
    positive: int
    neutral: int
    negative: int
    anger: int
    happiness: int
    core: str
    language_mix: LanguageMix
    speaker_results: List[SpeakerResult]

@app.post("/analyze")
def analyze_transcript(request: AnalyzeRequest):
    transcript_text = "\n".join(
        [f"{segment.speaker}: {segment.text}" for segment in request.segments]
    )

    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You analyze conversations for a stylized emotional report app called Scribed. "
                    "Return a structured sentiment report. "
                    "Percentages should be integers from 0 to 100. "
                    "positive, neutral, and negative should add up to 100. "
                    "language_mix should estimate English, Arabic, and mixed language usage. "
                    "core should be one dramatic one-word category such as TRANSCENDENT, STASIS, VOID, CHAOS, or SERENITY."
                )
            },
            {
                "role": "user",
                "content": transcript_text
            }
        ],
        response_format=SentimentReport,
    )

    report = completion.choices[0].message.parsed

    return report.model_dump()

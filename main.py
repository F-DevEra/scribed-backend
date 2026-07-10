from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import shutil
import os
import re
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
    start: float | None = None
    end: float | None = None


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

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
ENGLISH_RE = re.compile(r"[A-Za-z]")
def count_words(text: str) -> int:
    words = text.split()
    return max(len(words), 1)


def classify_segment_language(text: str) -> str:
    has_arabic = bool(ARABIC_RE.search(text))
    has_english = bool(ENGLISH_RE.search(text))

    if has_arabic and has_english:
        return "mixed"
    elif has_arabic:
        return "arabic"
    elif has_english:
        return "english"
    else:
        return "english"


def calculate_language_mix(segments: List[Segment]) -> LanguageMix:
    totals = {
        "english": 0,
        "arabic": 0,
        "mixed": 0
    }

    total_words = 0

    for segment in segments:
        word_count = count_words(segment.text)
        language = classify_segment_language(segment.text)

        totals[language] += word_count
        total_words += word_count

    if total_words == 0:
        return LanguageMix(english=0, arabic=0, mixed=0)

    english = round((totals["english"] / total_words) * 100)
    arabic = round((totals["arabic"] / total_words) * 100)

    # Ensures the total is exactly 100
    mixed = 100 - english - arabic

    return LanguageMix(
        english=english,
        arabic=arabic,
        mixed=mixed
    )
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
                    "Do not worry about calculating language percentages; they will be calculated separately. "
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
    report.language_mix = calculate_language_mix(request.segments)

    return report.model_dump()

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
import logging
import os
import re
import uuid


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scribed-backend")

app = FastAPI()
client = OpenAI()


MAX_AUDIO_SIZE = 20 * 1024 * 1024  # 20 MB

ALLOWED_EXTENSIONS = {
    ".m4a",
    ".mp3",
    ".wav",
    ".webm",
    ".mp4",
    ".mpeg"
}


@app.get("/health")
def health():
    return {
        "status": "backend working",
        "message": "Scribed backend is alive"
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    original_name = file.filename or ""
    extension = os.path.splitext(original_name)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio file type."
        )

    temp_filename = f"temp_{uuid.uuid4()}{extension}"

    try:
        file_size = 0

        with open(temp_filename, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                file_size += len(chunk)

                if file_size > MAX_AUDIO_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Audio file is too large."
                    )

                buffer.write(chunk)

        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty."
            )

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
            "filename": original_name,
            "segments": segments
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Transcription failed")

        raise HTTPException(
            status_code=500,
            detail="Transcription could not be completed."
        )

    finally:
        await file.close()

        if os.path.exists(temp_filename):
            os.remove(temp_filename)


class Segment(BaseModel):
    speaker: str
    text: str
    start: Optional[float] = None
    end: Optional[float] = None


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


ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"
)

ENGLISH_RE = re.compile(r"[A-Za-z]")


def classify_segment_language(text: str) -> str:
    has_arabic = bool(ARABIC_RE.search(text))
    has_english = bool(ENGLISH_RE.search(text))

    if has_arabic and has_english:
        return "mixed"

    if has_arabic:
        return "arabic"

    return "english"


def count_words(text: str) -> int:
    return len(text.split())


def calculate_language_mix(
    segments: List[Segment]
) -> LanguageMix:
    totals = {
        "english": 0,
        "arabic": 0,
        "mixed": 0
    }

    total_words = 0

    for segment in segments:
        word_count = count_words(segment.text)

        if word_count == 0:
            continue

        language = classify_segment_language(segment.text)

        totals[language] += word_count
        total_words += word_count

    if total_words == 0:
        return LanguageMix(
            english=0,
            arabic=0,
            mixed=0
        )

    english = round(
        (totals["english"] / total_words) * 100
    )

    arabic = round(
        (totals["arabic"] / total_words) * 100
    )

    mixed = 100 - english - arabic

    return LanguageMix(
        english=english,
        arabic=arabic,
        mixed=mixed
    )


@app.post("/analyze")
def analyze_transcript(request: AnalyzeRequest):
    if not request.segments:
        raise HTTPException(
            status_code=400,
            detail="No transcript segments were provided."
        )

    transcript_text = "\n".join(
        f"{segment.speaker}: {segment.text.strip()}"
        for segment in request.segments
        if segment.text.strip()
    )

    if not transcript_text:
        raise HTTPException(
            status_code=400,
            detail="The transcript is empty."
        )

    try:
        completion = client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze conversations for a stylized "
                        "emotional report app called Scribed. Return "
                        "a structured sentiment report. "

                        "Positive, neutral, negative, anger, and "
                        "happiness must be integers from 0 to 100. "
                        "Positive, neutral, and negative must add "
                        "up to 100. "

                        "For each speaker, analyze their conversational "
                        "tone separately. The tone field must contain "
                        "1 to 3 precise, nuanced tone labels separated "
                        "by commas. Examples include joyful, playful, "
                        "affectionate, hopeful, confident, relieved, "
                        "calm, reflective, thoughtful, curious, "
                        "reserved, serious, tired, detached, anxious, "
                        "uncertain, hesitant, guarded, overwhelmed, "
                        "frustrated, irritated, angry, sad, "
                        "disappointed, vulnerable, or exhausted. "

                        "Choose tones based on wording, hesitation, "
                        "intensity, emotional context, and conversational "
                        "style. Do not reduce speaker tones to positive, "
                        "neutral, negative, happiness, or anger. Do not "
                        "give every speaker the same generic tone. Do "
                        "not invent extreme emotions without evidence. "

                        "Do not calculate language percentages because "
                        "the backend calculates them separately. "

                        "Core must be one dramatic one-word category "
                        "such as TRANSCENDENT, STASIS, VOID, CHAOS, "
                        "or SERENITY."
                    )
                },
                {
                    "role": "user",
                    "content": transcript_text
                }
            ],
            response_format=SentimentReport
        )

        report = completion.choices[0].message.parsed

        if report is None:
            raise RuntimeError(
                "OpenAI returned no parsed report"
            )

        report.language_mix = calculate_language_mix(
            request.segments
        )

        return report.model_dump()

    except HTTPException:
        raise

    except Exception:
        logger.exception("Analysis failed")

        raise HTTPException(
            status_code=500,
            detail="Sentiment analysis could not be completed."
        )

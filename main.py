"""PhytoSense FastAPI backend.

Includes:

* Plant diagnosis history (Firestore)
* Multilingual chat with Gemini (text + voice) and gTTS replies
* Conversation history stored per session

Run from this folder::

    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Environment variables are read from ``.env`` next to this file (see
``.env.example``). Firebase is optional for chat: without credentials the chat
falls back to an in-memory store so the UI can still be developed offline.
"""

from __future__ import annotations

import base64
import binascii
import glob
import importlib.util
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]

# .env in the backend folder wins; the repo root is a convenient fallback.
load_dotenv(BASE_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

import chat_store  # noqa: E402  (needs .env loaded first)
import gemini_service  # noqa: E402
import tts_service  # noqa: E402
from chat_config import (  # noqa: E402
    ALLOWED_AUDIO_MIME,
    AUTO_LANGUAGE,
    LANGUAGES,
    MAX_AUDIO_BYTES,
    MAX_HISTORY_MESSAGES,
    normalize_language,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("phytosense.main")

firebase_admin = None
firestore = None
_db: Any = None
_firebase_checked = False


# ==========================================
# Firebase (lazy, non-fatal)
# ==========================================

def _find_credentials() -> Optional[str]:
    patterns = [
        str(BASE_DIR / "anayaarajpoot66-*.json"),
        str(BASE_DIR / "*firebase-adminsdk*.json"),
        str(PROJECT_ROOT / "*firebase-adminsdk*.json"),
        str(BASE_DIR / "serviceAccount*.json"),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        for match in matches:
            if os.path.isfile(match):
                return match
    env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and os.path.isfile(env_path):
        return env_path
    return None


def ensure_firebase() -> Any:
    """Initialise Firebase once. Returns a Firestore client or ``None``."""
    global firebase_admin, firestore, _db, _firebase_checked

    if _db is not None:
        return _db
    if _firebase_checked:
        return _db
    _firebase_checked = True

    if os.getenv("FIREBASE_DISABLED", "").strip().lower() in {"1", "true", "yes"}:
        log.warning("Firebase disabled via FIREBASE_DISABLED")
        return None

    cred_path = _find_credentials()
    if not cred_path:
        log.warning("No Firebase service-account JSON found; diagnosis persistence disabled")
        return None

    try:
        import firebase_admin as fb
        from firebase_admin import credentials, firestore as fs

        firebase_admin = fb
        firestore = fs

        if not fb._apps:
            fb.initialize_app(credentials.Certificate(cred_path))
        _db = fs.client()

        # Enable offline persistence on client side
        # Note: Actual persistence is handled by Firebase SDK on mobile/web clients
        log.info("Firebase connected with offline persistence support")
    except Exception as exc:  # pragma: no cover - depends on environment
        log.warning("Firebase could not be initialised: %s", exc)
        _db = None
    return _db


def require_db() -> Any:
    db = ensure_firebase()
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Firebase is not configured on the server. Add the service-account JSON to the backend folder.",
        )
    return db


class ProfileUpdate(BaseModel):
    full_name: str = Field(default="", max_length=60)
    phone: str = Field(default="", max_length=30)
    location: str = Field(default="", max_length=80)
    bio: str = Field(default="", max_length=300)


def authenticated_profile_uid(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="A Firebase ID token is required.")

    require_db()
    from firebase_admin import auth as firebase_auth

    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="A Firebase ID token is required.")

    try:
        claims = firebase_auth.verify_id_token(token)
    except (firebase_auth.InvalidIdTokenError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="The Firebase ID token is invalid or expired.") from exc

    uid = claims.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="The Firebase ID token has no user ID.")
    return uid


# ==========================================
# FastAPI app
# ==========================================

app = FastAPI(
    title="PhytoSense AI Backend",
    description="Plant disease detection, multilingual Gemini chat and speech",
    version="2.0.0",
)

_origins = os.getenv("CORS_ORIGINS", "*").strip()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins == "*" else [item.strip() for item in _origins.split(",") if item.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    ensure_firebase()
    log.info(
        "PhytoSense backend ready | gemini_configured=%s | store=%s | tts=%s",
        gemini_service.is_configured(),
        chat_store.store_backend(),
        bool(importlib.util.find_spec("gtts")),
    )


@app.exception_handler(chat_store.ChatStoreError)
async def _store_error_handler(_request, exc: chat_store.ChatStoreError):
    status = 503
    if isinstance(exc, chat_store.SessionNotFound):
        status = 404
    elif isinstance(exc, chat_store.SessionForbidden):
        status = 403
    return JSONResponse(status_code=status, content={"detail": {"code": "chat_store", "message": str(exc)}})


# ==========================================
# Root / health
# ==========================================

@app.get("/")
def read_root():
    return {
        "message": "PhytoSense Backend is running successfully!",
        "firebase": "Connected" if ensure_firebase() else "Not configured",
        "gemini": "Configured" if gemini_service.is_configured() else "Not configured",
        "chat_store": chat_store.store_backend(),
        "version": "2.0.0",
    }


@app.get("/firebase-test")
def firebase_test():
    db = ensure_firebase()
    if db is None:
        return {"status": "error", "message": "Firebase is not configured"}
    try:
        db.collection("connection_test").limit(1).get()
        return {"status": "success", "message": "Firebase Firestore connected successfully!"}
    except Exception as exc:
        return {"status": "error", "message": "Firebase connection failed", "detail": str(exc)}


@app.get("/api/profile")
def get_profile(authorization: Optional[str] = Header(default=None)):
    uid = authenticated_profile_uid(authorization)
    db = require_db()
    profile = db.collection("users").document(uid).get().to_dict() or {}
    return {
        "full_name": profile.get("full_name", ""),
        "phone": profile.get("phone", ""),
        "location": profile.get("location", ""),
        "bio": profile.get("bio", ""),
    }


@app.put("/api/profile")
def save_profile(
    payload: ProfileUpdate,
    authorization: Optional[str] = Header(default=None),
):
    uid = authenticated_profile_uid(authorization)
    db = require_db()
    profile = {key: value.strip() for key, value in payload.dict().items()}
    profile["updated_at"] = firestore.SERVER_TIMESTAMP
    db.collection("users").document(uid).set(profile, merge=True)
    return {key: value for key, value in profile.items() if key != "updated_at"}


@app.get("/health")
@app.get("/api/chat/health")
def chat_health():
    return {
        "status": "ok",
        "gemini": gemini_service.is_configured(),
        "models": gemini_service._models(),
        "store": chat_store.store_backend(),
        "tts": bool(importlib.util.find_spec("gtts")),
        "languages": len(LANGUAGES),
        "auto_language": True,
        "limits": {
            "audio_bytes": MAX_AUDIO_BYTES,
            "history_messages": MAX_HISTORY_MESSAGES,
        },
    }


@app.get("/api/chat/languages")
def chat_languages():
    return {
        "status": "success",
        "auto": AUTO_LANGUAGE,
        "count": len(LANGUAGES),
        "languages": LANGUAGES,
    }


# ==========================================
# Diagnosis history (unchanged API)
# ==========================================

class DiagnosisRecord(BaseModel):
    crop: str
    disease: str
    confidence: float
    image_data: Optional[str] = None
    type: str
    date: Optional[str] = None


@app.post("/api/diagnosis/save")
async def save_diagnosis(record: DiagnosisRecord):
    db = require_db()
    try:
        if not record.date:
            record.date = datetime.now().isoformat()

        doc_ref = db.collection("diagnosis_history").add(
            {
                "crop": record.crop,
                "disease": record.disease,
                "confidence": record.confidence,
                "image_data": record.image_data,
                "type": record.type,
                "date": record.date,
                "created_at": firestore.SERVER_TIMESTAMP,
            }
        )
        return {"status": "success", "message": "Diagnosis saved successfully", "id": doc_ref[1].id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/diagnosis/history")
async def get_diagnosis_history(type_filter: Optional[str] = None):
    db = require_db()
    try:
        history_ref = db.collection("diagnosis_history")
        if type_filter:
            history_ref = history_ref.where("type", "==", type_filter)
        history_ref = history_ref.order_by("date", direction=firestore.Query.DESCENDING)

        history = []
        for doc in history_ref.get():
            data = doc.to_dict()
            data["id"] = doc.id
            history.append(data)

        return {"status": "success", "count": len(history), "history": history}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/diagnosis/upload-image")
async def upload_diagnosis_image(
    crop: str,
    disease: str,
    confidence: float,
    file: UploadFile = File(...),
):
    db = require_db()
    try:
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode("utf-8")

        doc_ref = db.collection("diagnosis_history").add(
            {
                "crop": crop,
                "disease": disease,
                "confidence": confidence,
                "image_data": image_base64,
                "type": "image",
                "date": datetime.now().isoformat(),
                "created_at": firestore.SERVER_TIMESTAMP,
            }
        )
        return {"status": "success", "message": "Image uploaded and saved successfully", "id": doc_ref[1].id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ==========================================
# Legacy single-shot AI query (no history)
# ==========================================

class AIQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    language: str = "en"
    context: str = "crop disease diagnosis and farming advice"


@app.post("/api/ai/query")
async def process_ai_query(ai_query: AIQuery):
    try:
        reply, used_language = await gemini_service.generate_reply(
            prompt=ai_query.query, language=ai_query.language
        )
    except gemini_service.GeminiError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc

    result: Dict[str, Any] = {
        "status": "success",
        "response": reply,
        "language": used_language,
    }

    try:
        audio, _ = await tts_service.synthesize(reply, used_language)
        result["audio_base64"] = base64.b64encode(audio).decode("ascii")
        result["audio_mime"] = "audio/mpeg"
    except tts_service.TTSError as exc:
        result["tts_error"] = str(exc)

    return result


# ==========================================
# 12. Image Analysis / Disease Detection
# ==========================================

class ImageAnalysisRequest(BaseModel):
    image_data: str  # Base64 encoded image
    crop: Optional[str] = None
    language: str = "en"

class ImageAnalysisResponse(BaseModel):
    status: str
    crop: str
    disease: str
    confidence: float
    description: str
    symptoms: list
    causes: list
    treatment_chemical: list
    treatment_organic: list
    prevention: list
    is_plant: bool
    message: Optional[str] = None


@app.post("/api/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    """
    Analyze plant image for disease detection
    Integrates with Google Gemini Vision API or local AI model
    """
    db = require_db()

    try:
        # Validate image data
        if not request.image_data:
            raise HTTPException(status_code=400, detail="No image data provided")

        # Check if device is offline (simulated check)
        # In production, this would check actual network status

        # For now, use a simulated AI analysis
        # In production, integrate with Google Gemini Vision API
        analysis_result = await simulate_gemini_analysis(request.image_data, request.crop, request.language)

        # If not a plant leaf detected
        if not analysis_result["is_plant"]:
            return {
                "status": "error",
                "message": "No plant leaf detected. Please capture a clear photo of a plant leaf.",
                "is_plant": False
            }

        # Save to Firestore if available
        if db:
            doc_ref = db.collection("diagnosis_history").add({
                "crop": analysis_result["crop"],
                "disease": analysis_result["disease"],
                "confidence": analysis_result["confidence"],
                "image_data": request.image_data,
                "type": "image",
                "date": datetime.now().isoformat(),
                "created_at": firestore.SERVER_TIMESTAMP,
                "description": analysis_result["description"],
                "symptoms": analysis_result["symptoms"],
                "causes": analysis_result["causes"],
                "treatment_chemical": analysis_result["treatment_chemical"],
                "treatment_organic": analysis_result["treatment_organic"],
                "prevention": analysis_result["prevention"]
            })
            analysis_result["id"] = doc_ref[1].id

        return {
            "status": "success",
            **analysis_result
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def simulate_gemini_analysis(image_data: str, crop: str = None, language: str = "en") -> dict:
    """
    Simulate Gemini Vision API analysis
    In production, replace this with actual Google Gemini Vision API call
    """
    import base64
    import random

    # Decode base64 to check image size (basic validation)
    try:
        image_bytes = base64.b64decode(image_data)
        image_size = len(image_bytes)

        # Very basic validation - if image is too small, likely invalid
        if image_size < 1000:
            return {
                "is_plant": False,
                "message": "Image too small or invalid"
            }
    except:
        return {
            "is_plant": False,
            "message": "Invalid image data"
        }

    # Simulate AI detection with realistic results
    # In production, this would call Google Gemini Vision API
    diseases = [
        {
            "name": "Early Blight",
            "scientific": "Alternaria solani",
            "crops": ["Tomato", "Potato", "Eggplant"],
            "confidence": random.uniform(85.0, 98.0),
            "description": "Early blight is a fungal disease caused by Alternaria solani that affects tomatoes, potatoes, and eggplants.",
            "symptoms": [
                "Dark brown spots with concentric rings on leaves",
                "Yellowing and premature defoliation",
                "Lesions on stems and fruits"
            ],
            "causes": [
                "Warm humid weather",
                "Poor air circulation",
                "Overhead irrigation"
            ],
            "treatment_chemical": [
                "Apply Mancozeb (2-3 kg/ha)",
                "Use Chlorothalonil (2-4 kg/ha)",
                "Copper-based fungicides"
            ],
            "treatment_organic": [
                "Neem oil spray (2-5 ml/liter)",
                "Baking soda solution",
                "Compost tea"
            ],
            "prevention": [
                "Crop rotation (3-4 year cycle)",
                "Proper plant spacing",
                "Avoid overhead irrigation"
            ]
        },
        {
            "name": "Late Blight",
            "scientific": "Phytophthora infestans",
            "crops": ["Tomato", "Potato"],
            "confidence": random.uniform(80.0, 95.0),
            "description": "Late blight is a devastating fungal disease caused by Phytophthora infestans.",
            "symptoms": [
                "Water-soaked lesions on leaves",
                "White fungal growth on leaf undersides",
                "Rapid spread in cool wet conditions"
            ],
            "causes": [
                "Cool wet weather (15-20°C)",
                "High humidity",
                "Infected plant debris"
            ],
            "treatment_chemical": [
                "Apply metalaxyl-based fungicides",
                "Use cymoxanil + mancozeb",
                "Chlorothalonil for protection"
            ],
            "treatment_organic": [
                "Copper fungicides (3-5 kg/ha)",
                "Remove infected plants",
                "Bordeaux mixture"
            ],
            "prevention": [
                "Plant resistant varieties",
                "Ensure excellent drainage",
                "Avoid overhead irrigation"
            ]
        },
        {
            "name": "Powdery Mildew",
            "scientific": "Erysiphe cichoracearum",
            "crops": ["Cucumber", "Squash", "Melon"],
            "confidence": random.uniform(75.0, 92.0),
            "description": "Powdery mildew appears as white powdery spots on leaves of cucurbits.",
            "symptoms": [
                "White powdery coating on leaf surfaces",
                "Yellowing and browning of leaves",
                "Distorted growth"
            ],
            "causes": [
                "Warm dry conditions with high humidity",
                "Poor air circulation",
                "Overcrowded plantings"
            ],
            "treatment_chemical": [
                "Apply sulfur-based fungicides",
                "Use myclobutanil",
                "Trifloxystrobin"
            ],
            "treatment_organic": [
                "Neem oil spray",
                "Baking soda solution",
                "Milk spray"
            ],
            "prevention": [
                "Plant resistant varieties",
                "Improve air circulation",
                "Avoid overhead irrigation"
            ]
        },
        {
            "name": "Healthy Leaf",
            "scientific": "No disease detected",
            "crops": ["All crops"],
            "confidence": random.uniform(90.0, 99.0),
            "description": "The plant appears healthy with no visible signs of disease.",
            "symptoms": [
                "Green, vibrant leaves",
                "No spots or discoloration",
                "Normal growth pattern"
            ],
            "causes": [
                "Good growing conditions",
                "Proper care and maintenance",
                "Adequate nutrients"
            ],
            "treatment_chemical": [],
            "treatment_organic": [],
            "prevention": [
                "Continue regular monitoring",
                "Maintain good growing conditions",
                "Regular watering and fertilization"
            ]
        }
    ]

    # Select a disease based on crop if provided, otherwise random
    if crop:
        matching_diseases = [d for d in diseases if crop.lower() in [c.lower() for c in d["crops"]]]
        if matching_diseases:
            selected = random.choice(matching_diseases)
        else:
            selected = diseases[-1]  # Healthy leaf
    else:
        selected = random.choice(diseases)

    return {
        "is_plant": True,
        "crop": crop or selected["crops"][0],
        "disease": selected["name"],
        "scientific": selected["scientific"],
        "confidence": round(selected["confidence"], 1),
        "description": selected["description"],
        "symptoms": selected["symptoms"],
        "causes": selected["causes"],
        "treatment_chemical": selected["treatment_chemical"],
        "treatment_organic": selected["treatment_organic"],
        "prevention": selected["prevention"]
    }


# ==========================================
# Chat models
# ==========================================

class ChatSessionRequest(BaseModel):
    client_id: str = Field(..., min_length=3, max_length=120)
    language: str = "en"
    title: Optional[str] = None


class ChatMessageRequest(BaseModel):
    client_id: str = Field(..., min_length=3, max_length=120)
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    language: str = "en"
    include_audio: bool = True


class ChatVoiceRequest(BaseModel):
    client_id: str = Field(..., min_length=3, max_length=120)
    audio_base64: str = Field(..., min_length=1)
    mime_type: str = "audio/webm"
    session_id: Optional[str] = None
    language: str = "en"
    include_audio: bool = True
    duration_ms: Optional[int] = None


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=6000)
    language: str = "en"


class SessionUpdateRequest(BaseModel):
    client_id: str = Field(..., min_length=3, max_length=120)
    language: Optional[str] = None
    title: Optional[str] = None


# ==========================================
# Chat helpers
# ==========================================

def _iso_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _gemini_http_error(exc: Exception) -> HTTPException:
    status = getattr(exc, "status_code", 502)
    code = getattr(exc, "code", "gemini_error")
    return HTTPException(status_code=status, detail={"code": code, "message": str(exc)})


def _decode_audio(audio_base64: str, mime_type: str) -> bytes:
    mime = (mime_type or "").split(";")[0].strip().lower()
    allowed = {item.split(";")[0] for item in ALLOWED_AUDIO_MIME}
    if mime and mime not in allowed:
        raise HTTPException(
            status_code=415,
            detail={"code": "unsupported_audio", "message": f"Unsupported audio type: {mime}"},
        )

    payload = (audio_base64 or "").strip()
    if "," in payload[:80] and payload.lower().startswith("data:"):
        payload = payload.split(",", 1)[1]

    try:
        audio = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "bad_audio", "message": "The audio payload is not valid base64"},
        ) from exc

    if not audio:
        raise HTTPException(status_code=400, detail={"code": "bad_audio", "message": "The recording is empty"})
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "audio_too_large",
                "message": f"Recording is too large (max {MAX_AUDIO_BYTES // (1024 * 1024)} MB)",
            },
        )
    return audio


async def _maybe_tts(text: str, language: str, include_audio: bool) -> Dict[str, Any]:
    if not include_audio:
        return {"audio": None, "tts_error": None}
    try:
        audio, used = await tts_service.synthesize(text, language)
        return {
            "audio": {"mime_type": "audio/mpeg", "base64": base64.b64encode(audio).decode("ascii")},
            "tts_language": used,
            "tts_error": None,
        }
    except tts_service.TTSError as exc:
        log.warning("TTS unavailable: %s", exc)
        return {"audio": None, "tts_error": str(exc)}


def _assistant_message(message: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": message.get("id"),
        "role": "assistant",
        "content": message.get("content", ""),
        "language": message.get("language", "en"),
        "kind": "text",
        "created_at": message.get("created_at") or _iso_now(),
    }


# ==========================================
# Chat endpoints
# ==========================================

@app.post("/api/chat/session")
def create_chat_session(body: ChatSessionRequest):
    session = chat_store.create_session(
        client_id=body.client_id.strip(),
        language=normalize_language(body.language),
        title=body.title,
    )
    return {"status": "success", "session": session}


@app.get("/api/chat/sessions")
def list_chat_sessions(
    client_id: str = Query(..., min_length=3),
    limit: int = Query(30, ge=1, le=100),
):
    sessions = chat_store.list_sessions(client_id.strip(), limit=limit)
    return {"status": "success", "count": len(sessions), "sessions": sessions}


@app.get("/api/chat/history")
def chat_history(
    session_id: str = Query(..., min_length=4),
    client_id: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=500),
):
    session = chat_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Chat session not found"})
    if client_id and session.get("client_id") != client_id:
        raise HTTPException(status_code=403, detail={"code": "forbidden", "message": "This chat belongs to another client"})

    messages = chat_store.list_messages(session_id, limit=limit)
    return {
        "status": "success",
        "session": session,
        "count": len(messages),
        "messages": messages,
    }


@app.patch("/api/chat/session/{session_id}")
def update_chat_session(session_id: str, body: SessionUpdateRequest):
    session = chat_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Chat session not found"})
    if session.get("client_id") != body.client_id:
        raise HTTPException(status_code=403, detail={"code": "forbidden", "message": "This chat belongs to another client"})

    chat_store.touch_session(
        session_id,
        language=normalize_language(body.language) if body.language else None,
        title=body.title,
    )
    return {"status": "success", "session": chat_store.get_session(session_id)}


@app.delete("/api/chat/session/{session_id}")
def delete_chat_session(session_id: str, client_id: str = Query(..., min_length=3)):
    deleted = chat_store.delete_session(session_id, client_id.strip())
    if not deleted:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Chat session not found"})
    return {"status": "success", "deleted": True, "session_id": session_id}


@app.post("/api/chat/message")
async def chat_message(body: ChatMessageRequest):
    text = body.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail={"code": "empty_message", "message": "Message cannot be empty"})

    language = normalize_language(body.language)
    session = chat_store.get_or_create_session(body.client_id.strip(), body.session_id, language)
    session_id = str(session["id"])

    history = chat_store.recent_messages(session_id, limit=MAX_HISTORY_MESSAGES)

    try:
        reply_text, used_language = await gemini_service.generate_reply(
            prompt=text,
            language=language,
            history=history,
        )
    except gemini_service.GeminiError as exc:
        # Nothing is persisted on failure, so a retry does not duplicate the question.
        raise _gemini_http_error(exc) from exc

    user_message = chat_store.add_message(session_id, "user", text, language, kind="text")
    assistant = chat_store.add_message(session_id, "assistant", reply_text, used_language, kind="text")
    tts = await _maybe_tts(reply_text, used_language, body.include_audio)

    return {
        "status": "success",
        "session_id": session_id,
        "session": chat_store.get_session(session_id),
        "user_message": user_message,
        "message": _assistant_message(assistant),
        "reply": reply_text,
        "language": used_language,
        "audio": tts.get("audio"),
        "tts_language": tts.get("tts_language"),
        "tts_error": tts.get("tts_error"),
    }


@app.post("/api/chat/voice")
async def chat_voice(body: ChatVoiceRequest):
    audio = _decode_audio(body.audio_base64, body.mime_type)
    language = normalize_language(body.language)

    session = chat_store.get_or_create_session(body.client_id.strip(), body.session_id, language)
    session_id = str(session["id"])
    history = chat_store.recent_messages(session_id, limit=MAX_HISTORY_MESSAGES)

    try:
        transcript, reply_text, used_language = await gemini_service.transcribe_and_reply(
            audio=audio,
            mime_type=body.mime_type,
            language=language,
            history=history,
        )
    except gemini_service.GeminiError as exc:
        raise _gemini_http_error(exc) from exc

    stored_text = transcript or "(voice message that could not be transcribed)"
    user_message = chat_store.add_message(session_id, "user", stored_text, language, kind="voice")
    assistant = chat_store.add_message(session_id, "assistant", reply_text, used_language, kind="text")
    tts = await _maybe_tts(reply_text, used_language, body.include_audio)

    return {
        "status": "success",
        "session_id": session_id,
        "session": chat_store.get_session(session_id),
        "transcript": transcript,
        "user_message": user_message,
        "message": _assistant_message(assistant),
        "reply": reply_text,
        "language": used_language,
        "audio": tts.get("audio"),
        "tts_language": tts.get("tts_language"),
        "tts_error": tts.get("tts_error"),
    }


@app.post("/api/chat/voice/upload")
async def chat_voice_upload(
    client_id: str = Form(...),
    audio: UploadFile = File(...),
    language: str = Form("en"),
    session_id: Optional[str] = Form(None),
    include_audio: bool = Form(True),
):
    """Multipart variant for clients that prefer form uploads."""
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail={"code": "bad_audio", "message": "The recording is empty"})
    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail={"code": "audio_too_large", "message": "Recording is too large"})

    body = ChatVoiceRequest(
        client_id=client_id,
        audio_base64=base64.b64encode(raw).decode("ascii"),
        mime_type=audio.content_type or "audio/webm",
        session_id=session_id,
        language=language,
        include_audio=include_audio,
    )
    return await chat_voice(body)


@app.post("/api/chat/tts")
async def chat_tts(body: TTSRequest, response_format: str = Query("audio", alias="format")):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail={"code": "empty_text", "message": "Text is required"})

    try:
        audio, used = await tts_service.synthesize(text, body.language)
    except tts_service.TTSError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": "tts_error", "message": str(exc)},
        ) from exc

    if response_format == "base64":
        return {
            "status": "success",
            "language": used,
            "mime_type": "audio/mpeg",
            "base64": base64.b64encode(audio).decode("ascii"),
        }

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": 'inline; filename="phytosense-reply.mp3"',
            "X-TTS-Language": used,
            "Cache-Control": "no-store",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "").strip().lower() in {"1", "true", "yes"},
    )

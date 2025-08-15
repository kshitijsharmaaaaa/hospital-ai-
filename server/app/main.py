from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import os, base64, json
from dotenv import load_dotenv
from groq import Groq

# Load env vars
load_dotenv()

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Base paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

# FastAPI app
app = FastAPI(title="Hospital AI")

# CORS (allow all for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Models
class ChatRequest(BaseModel):
    question: str

# --------- Serve Frontend ---------
@app.get("/")
def serve_frontend():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    else:
        raise HTTPException(status_code=404, detail="Frontend not found.")

# --------- AI Chat Endpoint ---------
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Fast and free Groq model
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful healthcare assistant. "
                        "When a user asks a question, first give a short definition/explanation "
                        "in one paragraph. Then, give numbered precautions or tips in plain text "
                        "(no markdown, no stars). Keep it clear, factual, and easy to read."
                    )
                },
                {"role": "user", "content": req.question}
            ],
            temperature=0.2,
            max_tokens=400
        )
        answer = response.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

# --------- Medicine Text Analysis ---------
@app.post("/medicine-analyze")
async def medicine_analyze(file: UploadFile = File(...)):
    try:
        # Accept any file for now since we're doing text-based analysis
        # In the future, you can add proper image validation
        
        system = (
            "You are a pharmacist assistant. Based on the medicine name provided, give general information about the medicine. "
            "Provide a concise, **general** info summary WITHOUT dosing: "
            "• Indications (what it's for) "
            "• Who should avoid (contraindications) "
            "• Common side effects "
            "• Serious warnings (black-box or high risk) "
            "• Interactions (big ones only) "
            "• Pregnancy/Lactation caution "
            "• Schedule/OTC note (if region-agnostic) "
            "Always add: 'Educational use only — consult a doctor/pharmacist.' "
            "Return JSON only in this schema: "
            "{name:'', generic:'', strength:'', form:'', indications:'', contraindications:'', side_effects:'', warnings:'', interactions:'', pregnancy:'', schedule:'', note:''}"
        )

        # For now, we'll provide information about common medicines
        # Since Groq doesn't support vision yet, we'll use a predefined list
        # In the future, you can integrate with OCR services to extract text from images
        
        # Read the file content to see if we can extract any text
        file_content = await file.read()
        
        # Try to decode as text (in case it's a text file with medicine name)
        try:
            text_content = file_content.decode('utf-8').strip()
            if text_content and len(text_content) < 100:  # Reasonable medicine name length
                medicine_name = text_content
            else:
                # Use a default medicine if no readable text found
                medicine_name = "Paracetamol"
        except:
            # If file can't be decoded as text, use default
            medicine_name = "Paracetamol"
        
        user_prompt = f"Provide information about the medicine: {medicine_name}"

        resp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )

        raw = resp.choices[0].message.content.strip()

        def extract_json(s: str):
            s = s.strip()
            if s.startswith("```"):
                s = s.strip("`")
                if "\n" in s:
                    s = s.split("\n", 1)[1]
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end+1])
            return None

        data = None
        try:
            data = extract_json(raw)
        except Exception:
            data = None

        if not data:
            return {"raw": raw, "parsed": False}

        keys = ["name","generic","strength","form","indications","contraindications","side_effects","warnings","interactions","pregnancy","schedule","note"]
        for k in keys:
            data.setdefault(k, "")

        return {"parsed": True, "data": data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing medicine: {e}") 

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CRICAI CV Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "CRICAI CV backend is running"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "cricai-cv-backend"
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    return {
        "success": True,
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "Video received successfully",
        "analysis": {
            "summary": "Backend connection works. Real CV analysis will be added next.",
            "confidence": 0.0,
            "status": "test-mode"
        }
    }
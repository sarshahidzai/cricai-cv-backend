from fastapi import FastAPI, UploadFile, File, Form
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
async def analyze(file: UploadFile = File(...), analysis_type: str = Form("bowling")):
    try:
        import tempfile
        import os

        suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            video_bytes = await file.read()
            tmp.write(video_bytes)
            temp_video_path = tmp.name

        extracted = extract_pose_data(temp_video_path, analysis_type)

        pose_data = extracted.get("pose_data", [])
        frames_with_pose = extracted.get("frames_with_pose", 0)
        frame_count = extracted.get("frame_count", len(pose_data))
        pose_confidence = extracted.get("pose_confidence", 0.0)

        return {
            "success": True,
            "filename": file.filename,
            "content_type": file.content_type,
            "message": "Video analyzed successfully",
            "analysis_mode": "full-pose-detection",
            "frame_count": frame_count,
            "frames_with_pose": frames_with_pose,
            "pose_data": pose_data,
            "pose_confidence": pose_confidence,
            "analysis": extracted.get("analysis", {
                "summary": "CV analysis completed",
                "confidence": pose_confidence,
                "status": "complete"
            })
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Analysis failed: {str(e)}",
            "analysis_mode": "full-pose-detection",
            "frame_count": 0,
            "frames_with_pose": 0,
            "pose_data": [],
            "pose_confidence": 0.0,
            "analysis": {
                "summary": f"Analysis failed: {str(e)}",
                "confidence": 0.0,
                "status": "error"
            }
        }

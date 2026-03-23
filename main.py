from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import cv2
import numpy as np
import requests
import mediapipe as mp

app = FastAPI(title="CRICAI CV Backend", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

class AnalyzeUrlRequest(BaseModel):
    video_url: str
    analysis_type: str = "batting"

@app.get("/")
def root():
    return {"message": "CRICAI backend running"}

@app.get("/health")
def health():
    return {"status": "ok"}

def extract_pose_data(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    pose_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            pose_frames += 1

    cap.release()

    return {
        "frame_count": frame_count,
        "frames_with_pose": pose_frames
    }

def run_analysis(video_path, analysis_type):
    data = extract_pose_data(video_path)

    confidence = 0
    if data["frame_count"] > 0:
        confidence = int((data["frames_with_pose"] / data["frame_count"]) * 100)

    result = {
        "success": True,
        "analysisAvailable": True,
        "analysis_type": analysis_type,
        "confidence_score": confidence,
        "frame_count": data["frame_count"],
        "frames_with_pose": data["frames_with_pose"]
    }

    if analysis_type == "batting":
        result["shot_type"] = "cover drive"
    else:
        result["delivery_type"] = "fast bowling"

    return result

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), analysis_type: str = Form("batting")):
    temp_path = None
    try:
        suffix = ".mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            video_bytes = await file.read()
            tmp.write(video_bytes)
            temp_path = tmp.name

        result = run_analysis(temp_path, analysis_type)
        return result

    except Exception as e:
        return {"success": False, "message": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/analyze-url")
async def analyze_url(body: AnalyzeUrlRequest):
    temp_path = None
    try:
        r = requests.get(body.video_url, timeout=60)
        r.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(r.content)
            temp_path = tmp.name

        result = run_analysis(temp_path, body.analysis_type)
        return result

    except Exception as e:
        return {"success": False, "message": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
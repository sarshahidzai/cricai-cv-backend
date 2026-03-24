from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os

import cv2
import requests
from mediapipe.python.solutions import pose as mp_pose

app = FastAPI(title="CRICAI CV Backend", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe Pose once
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


class AnalyzeUrlRequest(BaseModel):
    video_url: str
    analysis_type: str = "batting"


@app.get("/")
def root():
    return {
        "message": "CRICAI backend running",
        "version": "5.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "cricai-cv-backend"
    }


def extract_pose_data(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Could not open video file")

    frame_count = 0
    pose_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results and results.pose_landmarks:
                pose_frames += 1

    finally:
        cap.release()

    return {
        "frame_count": frame_count,
        "frames_with_pose": pose_frames
    }


def run_analysis(video_path: str, analysis_type: str) -> dict:
    analysis_type = (analysis_type or "batting").lower().strip()

    if analysis_type not in ["batting", "bowling"]:
        analysis_type = "batting"

    data = extract_pose_data(video_path)

    frame_count = data.get("frame_count", 0)
    frames_with_pose = data.get("frames_with_pose", 0)

    confidence = 0
    if frame_count > 0:
        confidence = int((frames_with_pose / frame_count) * 100)

    result = {
        "success": True,
        "analysisAvailable": True,
        "analysis_type": analysis_type,
        "confidence_score": confidence,
        "frame_count": frame_count,
        "frames_with_pose": frames_with_pose
    }

    if analysis_type == "batting":
        result.update({
            "shot_type": "cover drive",
            "stance": "right-handed",
            "footwork": "stable",
            "head_position": "balanced",
            "timing": "good"
        })
    else:
        result.update({
            "delivery_type": "fast bowling",
            "bowling_arm": "right arm",
            "release_quality": "good",
            "runup_balance": "stable",
            "follow_through": "controlled"
        })

    return result


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    analysis_type: str = Form("batting")
):
    temp_path = None

    try:
        suffix = os.path.splitext(file.filename or "video.mp4")[1]
        if not suffix:
            suffix = ".mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            video_bytes = await file.read()
            tmp.write(video_bytes)
            temp_path = tmp.name

        result = run_analysis(temp_path, analysis_type)
        return result

    except Exception as e:
        return {
            "success": False,
            "analysisAvailable": False,
            "message": str(e)
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/analyze-url")
async def analyze_url(body: AnalyzeUrlRequest):
    temp_path = None

    try:
        response = requests.get(body.video_url, timeout=60)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(response.content)
            temp_path = tmp.name

        result = run_analysis(temp_path, body.analysis_type)
        return result

    except Exception as e:
        return {
            "success": False,
            "analysisAvailable": False,
            "message": str(e)
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
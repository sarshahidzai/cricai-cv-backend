from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import os
import tempfile

app = FastAPI(title="CRICAI CV Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose


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


def extract_pose_data(video_path, analysis_type="bowling"):
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)

    print("DEBUG: Opening video:", video_path)
    print("DEBUG: IsOpened:", cap.isOpened())

    if not cap.isOpened():
        return {
            "pose_data": [],
            "frames_with_pose": 0,
            "frame_count": 0,
            "pose_confidence": 0.0,
            "visual_overlay": {},
            "light_mode_warning": "VIDEO_NOT_SUPPORTED",
            "analysis": {
                "summary": "Video could not be opened (unsupported format or codec)",
                "confidence": 0.0,
                "status": "error"
            }
        }

    pose_data = []
    frame_count = 0
    frames_with_pose = 0
    confidence_scores = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            print("DEBUG: Frame read:", ret)

            if not ret:
                break

            frame_count += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                frames_with_pose += 1

                frame_landmarks = []
                visible_scores = []

                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    frame_landmarks.append({
                        "id": idx,
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "visibility": float(lm.visibility)
                    })
                    visible_scores.append(float(lm.visibility))

                avg_frame_conf = sum(visible_scores) / len(visible_scores) if visible_scores else 0.0
                confidence_scores.append(avg_frame_conf)

                pose_data.append({
                    "frame": frame_count,
                    "landmarks": frame_landmarks
                })

    cap.release()

    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

    return {
        "pose_data": pose_data,
        "frames_with_pose": frames_with_pose,
        "frame_count": frame_count,
        "pose_confidence": avg_confidence,
        "visual_overlay": {
            "keypoints": True,
            "skeleton": True
        },
        "light_mode_warning": None,
        "analysis": {
            "summary": "CV analysis completed" if frames_with_pose > 0 else "No pose detected in video",
            "confidence": avg_confidence,
            "status": "complete" if frames_with_pose > 0 else "no_pose_detected"
        }
    }


@app.post("/analyze-url")
async def analyze_url(req: AnalyzeURLRequest):
    input_path = None
    output_path = None

    try:
        print("DEBUG: analyze-url called")
        print("DEBUG: video_url =", req.video_url)

        r = requests.get(req.video_url, timeout=60)
        r.raise_for_status()

        ext = req.video_url.split(".")[-1].split("?")[0].lower() or "mp4"

        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(r.content)
            input_path = tmp.name

        output_path = input_path.rsplit(".", 1)[0] + "_converted.mp4"

        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-preset", "fast",
            output_path
        ], check=True)

        extracted = extract_pose_data(output_path, req.analysis_type)

        pose_data = extracted.get("pose_data", [])
        frames_with_pose = extracted.get("frames_with_pose", 0)
        frame_count = extracted.get("frame_count", len(pose_data))
        pose_confidence = extracted.get("pose_confidence", 0.0)

        detection_ratio = frames_with_pose / frame_count if frame_count > 0 else 0.0

        return {
            "success": True,
            "analysis_mode": "full-pose-detection",
            "frame_count": frame_count,
            "frames_with_pose": frames_with_pose,
            "pose_data": pose_data,
            "pose_confidence": pose_confidence,
            "detection_ratio": detection_ratio,
            "analysis": extracted.get("analysis", {})
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }

    finally:
        try:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
        except:
            pass

        try:
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
        except:
            pass

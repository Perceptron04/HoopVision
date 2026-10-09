
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STUBS_DEFAULT_PATH = PROJECT_ROOT / "stubs"

PLAYER_DETECTOR_PATH = (
    PROJECT_ROOT / "models" / "player_detection.pt"
)

BALL_DETECTOR_PATH = (
    PROJECT_ROOT / "models" / "ball_detection.pt"
)

COURT_KEYPOINT_DETECTOR_PATH = (
    PROJECT_ROOT / "models" / "court_keypoint_detection.pt"
)

OUTPUT_VIDEO_PATH = (
    PROJECT_ROOT / "output_videos" / "output_video.mp4"
)
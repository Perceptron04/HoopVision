from ultralytics import YOLO
import supervision as sv
import sys

sys.path.append('../')

from utils import read_stub, save_stub


class PlayerTracker:
    """
    A class that handles player detection and tracking using YOLO and ByteTrack.
    """

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        """
        Detect players in a sequence of frames using batch processing.
        """

        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i:i + batch_size],
                conf=0.5
            )

            detections += detections_batch

        return detections

    def get_object_tracks(
        self,
        frames,
        read_from_stub=False,
        stub_path=None
    ):
        """
        Get player tracking results for a sequence of frames.

        Each player track contains:
            - bbox
            - confidence
        """

        tracks = read_stub(read_from_stub, stub_path)

        # Only use cached tracks if they contain confidence scores.
        if tracks is not None and len(tracks) == len(frames):

            cache_has_confidence = True

            for frame_tracks in tracks:
                for player in frame_tracks.values():
                    if "confidence" not in player:
                        cache_has_confidence = False
                        break

                if not cache_has_confidence:
                    break

            if cache_has_confidence:
                return tracks

        # Run detection again if old cache does not contain confidence.
        detections = self.detect_frames(frames)

        tracks = []

        for frame_num, detection in enumerate(detections):

            cls_names = detection.names
            cls_names_inv = {
                v: k for k, v in cls_names.items()
            }

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(
                detection
            )

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks.append({})

            for frame_detection in detection_with_tracks:

                bbox = frame_detection[0].tolist()
                confidence = frame_detection[2]
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['Player']:

                    tracks[frame_num][track_id] = {
                        "bbox": bbox,
                        "confidence": float(confidence)
                    }

        save_stub(stub_path, tracks)

        return tracks
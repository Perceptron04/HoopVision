from ultralytics import YOLO
import supervision as sv

from utils import read_stub, save_stub


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames, batch_size=20):
        """
        Run player detection on batches of frames.
        """
        detections = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]

            batch_detections = self.model.predict(
                batch,
                conf=0.5,
                verbose=False
            )

            detections.extend(batch_detections)

        return detections

    def _tracks_from_detections(self, detections):
        """
        Convert YOLO detections into tracked player dictionaries.
        ByteTrack is kept alive across frames to preserve track IDs.
        """
        tracks = []

        for detection in detections:
            class_names = detection.names
            class_names_inv = {
                name: class_id
                for class_id, name in class_names.items()
            }

            if "Player" not in class_names_inv:
                raise ValueError(
                    "The player model has no class named 'Player'. "
                    f"Available classes: {list(class_names.values())}"
                )

            player_class_id = class_names_inv["Player"]

            supervision_detections = (
                sv.Detections.from_ultralytics(detection)
            )

            tracked_detections = (
                self.tracker.update_with_detections(
                    supervision_detections
                )
            )

            frame_tracks = {}

            for item in tracked_detections:
                bbox = item[0].tolist()
                confidence = item[2]
                class_id = item[3]
                track_id = item[4]

                if class_id != player_class_id or track_id is None:
                    continue

                frame_tracks[int(track_id)] = {
                    "bbox": bbox,
                    "confidence": float(confidence)
                }

            tracks.append(frame_tracks)

        return tracks

    def get_object_tracks(
        self,
        frames,
        read_from_stub=False,
        stub_path=None
    ):
        """
        Process a list of frames and return player tracks.

        This method preserves the existing interface used by main.py.
        It still requires frames to be held in memory.
        """
        cached_tracks = read_stub(read_from_stub, stub_path)

        if cached_tracks is not None and len(cached_tracks) == len(frames):
            cache_has_confidence = all(
                "confidence" in player
                for frame_tracks in cached_tracks
                for player in frame_tracks.values()
            )

            if cache_has_confidence:
                return cached_tracks

        detections = self.detect_frames(frames)
        tracks = self._tracks_from_detections(detections)

        save_stub(stub_path, tracks)

        return tracks
from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd

from utils import read_stub, save_stub


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, batch_size=20):
        """
        Run ball detection in batches to limit inference memory usage.
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

    def get_object_tracks(
        self,
        frames,
        read_from_stub=False,
        stub_path=None
    ):
        """
        Return the highest-confidence ball detection for each frame.
        """
        tracks = read_stub(read_from_stub, stub_path)

        if tracks is not None and len(tracks) == len(frames):
            return tracks

        detections = self.detect_frames(frames)
        tracks = []

        for detection in detections:
            class_names = detection.names
            class_names_inv = {
                name: class_id
                for class_id, name in class_names.items()
            }

            if "Ball" not in class_names_inv:
                raise ValueError(
                    "The ball model has no class named 'Ball'. "
                    f"Available classes: {list(class_names.values())}"
                )

            ball_class_id = class_names_inv["Ball"]

            supervision_detections = (
                sv.Detections.from_ultralytics(detection)
            )

            frame_tracks = {}
            chosen_bbox = None
            max_confidence = 0.0

            for item in supervision_detections:
                bbox = item[0].tolist()
                class_id = item[3]
                confidence = item[2]

                if (
                    class_id == ball_class_id
                    and confidence is not None
                    and confidence > max_confidence
                ):
                    chosen_bbox = bbox
                    max_confidence = float(confidence)

            if chosen_bbox is not None:
                frame_tracks[1] = {
                    "bbox": chosen_bbox,
                    "confidence": max_confidence
                }

            tracks.append(frame_tracks)

        save_stub(stub_path, tracks)

        return tracks

    def remove_wrong_detections(self, ball_positions):
        """
        Remove detections whose displacement exceeds the allowed threshold.
        """
        maximum_allowed_distance = 25
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_box = (
                ball_positions[i].get(1, {}).get("bbox", [])
            )

            if len(current_box) == 0:
                continue

            if last_good_frame_index == -1:
                last_good_frame_index = i
                continue

            last_good_box = (
                ball_positions[last_good_frame_index]
                .get(1, {})
                .get("bbox", [])
            )

            frame_gap = i - last_good_frame_index
            adjusted_max_distance = (
                maximum_allowed_distance * frame_gap
            )

            current_center = np.array(current_box[:2])
            last_good_center = np.array(last_good_box[:2])

            if (
                np.linalg.norm(current_center - last_good_center)
                > adjusted_max_distance
            ):
                ball_positions[i] = {}
            else:
                last_good_frame_index = i

        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolate missing ball bounding boxes across the video.
        """
        boxes = [
            frame.get(1, {}).get("bbox", [])
            for frame in ball_positions
        ]

        df = pd.DataFrame(
            [
                box if len(box) == 4 else [np.nan] * 4
                for box in boxes
            ],
            columns=["x1", "y1", "x2", "y2"]
        )

        df = df.interpolate().bfill()

        result = []

        for box in df.to_numpy().tolist():
            if any(pd.isna(value) for value in box):
                result.append({})
            else:
                result.append({
                    1: {"bbox": box}
                })

        return result
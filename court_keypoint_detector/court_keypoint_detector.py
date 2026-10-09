from ultralytics import YOLO

from utils import read_stub, save_stub


class CourtKeypointDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def get_court_keypoints(
        self,
        frames,
        read_from_stub=False,
        stub_path=None,
        batch_size=20
    ):
        """
        Detect court keypoints in batches and optionally use cached results.
        """
        court_keypoints = read_stub(read_from_stub, stub_path)

        if (
            court_keypoints is not None
            and len(court_keypoints) == len(frames)
        ):
            return court_keypoints

        court_keypoints = []

        for start in range(0, len(frames), batch_size):
            batch = frames[start:start + batch_size]

            detections_batch = self.model.predict(
                batch,
                conf=0.5,
                verbose=False
            )

            for detection in detections_batch:
                court_keypoints.append(detection.keypoints)

        save_stub(stub_path, court_keypoints)

        return court_keypoints

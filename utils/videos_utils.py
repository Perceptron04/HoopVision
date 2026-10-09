import os
import cv2


def get_video_properties(video_path):
    """
    Return the video's FPS, frame count, width, and height.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid video properties for: {video_path}"
        )

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def iter_video_frames(video_path):
    """
    Yield one video frame at a time instead of loading the entire video.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            yield frame
    finally:
        cap.release()


class StreamingVideoWriter:
    """
    Write frames incrementally without storing the output video in RAM.
    """

    def __init__(self, output_video_path, fps, width, height):
        output_dir = os.path.dirname(output_video_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.output_video_path = output_video_path

        # OpenCV's MPEG-4 Part 2 codec.
        # Use an output path ending in .mp4.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            (width, height)
        )

        if not self.writer.isOpened():
            raise RuntimeError(
                f"Could not create output video: {output_video_path}"
            )

        self.closed = False

    def write(self, frame):
        """
        Write a single annotated frame immediately to disk.
        """
        if self.closed:
            raise RuntimeError("Cannot write to a closed video writer.")

        self.writer.write(frame)

    def release(self):
        """
        Release the video writer.
        """
        if not self.closed:
            self.writer.release()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def read_video(video_path):
    """
    Read all frames into memory.

    Retained for compatibility with the existing pipeline.
    Use iter_video_frames() for memory-efficient sequential reading.
    """
    return list(iter_video_frames(video_path))


def save_video(output_video_frames, output_video_path, fps=24):
    """
    Save an existing list of frames to a video.

    Retained for compatibility. This function does not stream the
    input frames because they are already stored in a list.
    """
    if len(output_video_frames) == 0:
        raise ValueError("No frames available to save.")

    height, width = output_video_frames[0].shape[:2]

    with StreamingVideoWriter(
        output_video_path,
        fps,
        width,
        height
    ) as writer:
        for frame in output_video_frames:
            writer.write(frame)

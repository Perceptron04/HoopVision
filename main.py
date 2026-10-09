
import os
import argparse
from itertools import islice

import cv2

from utils import (
    get_video_properties,
    iter_video_frames,
    StreamingVideoWriter,
)

from trackers import PlayerTracker, BallTracker
from team_assigner import TeamAssigner
from court_keypoint_detector import CourtKeypointDetector
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator

from drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer,
    CourtKeypointDrawer,
    TeamBallControlDrawer,
    PassInterceptionDrawer,
    TacticalViewDrawer,
    SpeedAndDistanceDrawer,
)

from configs import (
    STUBS_DEFAULT_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Memory-efficient basketball video analysis"
    )

    parser.add_argument(
        "input_video",
        type=str,
        help="Path to the input video",
    )

    parser.add_argument(
        "--output_video",
        type=str,
        default=str(OUTPUT_VIDEO_PATH),
        help="Path to the output MP4 video",
    )

    parser.add_argument(
        "--stub_path",
        type=str,
        default=str(STUBS_DEFAULT_PATH),
        help="Path to the stub directory",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=30,
        help="Number of video frames processed per chunk",
    )

    return parser.parse_args()


def read_frame_batch(frame_iterator, batch_size):
    """Read at most batch_size frames into memory."""
    return list(islice(frame_iterator, batch_size))


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")

    if not os.path.isfile(args.input_video):
        raise FileNotFoundError(
            f"Input video not found: {args.input_video}"
        )

    properties = get_video_properties(args.input_video)

    fps = properties["fps"]
    width = properties["width"]
    height = properties["height"]

    print(f"Input video: {args.input_video}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Chunk size: {args.batch_size}")
    print("Starting analysis pass...")

    # ---------------------------------------------------------
    # 1. Initialize models and analyzers once.
    # ---------------------------------------------------------

    player_tracker = PlayerTracker(
        str(PLAYER_DETECTOR_PATH)
    )

    ball_tracker = BallTracker(
        str(BALL_DETECTOR_PATH)
    )

    court_keypoint_detector = CourtKeypointDetector(
        str(COURT_KEYPOINT_DETECTOR_PATH)
    )

    team_assigner = TeamAssigner()
    team_assigner.load_model()

    # Compact results are retained; decoded video frames are not.
    player_tracks = []
    ball_tracks = []
    court_keypoints_per_frame = []
    player_assignment = []

    # ---------------------------------------------------------
    # 2. First pass: analyze small batches of frames.
    # ---------------------------------------------------------

    frame_iterator = iter_video_frames(args.input_video)
    frame_number = 0

    try:
        while True:
            frames = read_frame_batch(
                frame_iterator,
                args.batch_size
            )

            if not frames:
                break

            batch_count = len(frames)
            batch_start = frame_number

            print(
                f"\rAnalyzing frames "
                f"{batch_start + 1}-"
                f"{batch_start + batch_count}",
                end="",
                flush=True,
            )

            # Disable cached stubs here. Existing whole-video stubs
            # cannot safely be reused as individual chunk results.
            batch_player_tracks = (
                player_tracker.get_object_tracks(
                    frames,
                    read_from_stub=False,
                    stub_path=None,
                )
            )

            batch_ball_tracks = (
                ball_tracker.get_object_tracks(
                    frames,
                    read_from_stub=False,
                    stub_path=None,
                )
            )

            batch_keypoints = (
                court_keypoint_detector.get_court_keypoints(
                    frames,
                    read_from_stub=False,
                    stub_path=None,
                    batch_size=min(20, args.batch_size),
                )
            )

            # Team assignment: load Fashion-CLIP once and preserve
            # the original 50-frame reset cadence across chunks.
            for local_index, (frame, frame_tracks) in enumerate(
                zip(frames, batch_player_tracks)
            ):
                global_index = batch_start + local_index

                if global_index % 50 == 0:
                    team_assigner.player_team_dict = {}

                frame_assignment = {}

                for player_id, player_data in frame_tracks.items():
                    team_id = team_assigner.get_player_team(
                        frame,
                        player_data["bbox"],
                        player_id,
                    )

                    frame_assignment[player_id] = team_id

                player_assignment.append(frame_assignment)

            player_tracks.extend(batch_player_tracks)
            ball_tracks.extend(batch_ball_tracks)
            court_keypoints_per_frame.extend(batch_keypoints)

            frame_number += batch_count

            # Allow this batch of full-resolution frames to be freed.
            del frames
            del batch_player_tracks
            del batch_ball_tracks
            del batch_keypoints

    finally:
        frame_iterator.close()

    if frame_number == 0:
        raise ValueError("The input video contains no readable frames.")

    print(f"\nAnalysis complete: {frame_number} frames")

    # ---------------------------------------------------------
    # 3. Compute sequence-level analytics.
    # These operate on compact per-frame results, not video frames.
    # ---------------------------------------------------------

    print("Calculating ball positions...")

    ball_tracks = ball_tracker.remove_wrong_detections(
        ball_tracks
    )

    ball_tracks = ball_tracker.interpolate_ball_positions(
        ball_tracks
    )

    print("Calculating possession...")

    ball_aquisition_detector = BallAquisitionDetector()

    ball_aquisition = (
        ball_aquisition_detector.detect_ball_possession(
            player_tracks,
            ball_tracks,
        )
    )

    print("Calculating passes and interceptions...")

    event_detector = PassAndInterceptionDetector()

    passes = event_detector.detect_passes(
        ball_aquisition,
        player_assignment,
    )

    interceptions = event_detector.detect_interceptions(
        ball_aquisition,
        player_assignment,
    )

    print("Calculating tactical positions...")

    tactical_view_converter = TacticalViewConverter(
        court_image_path="./images/basketball_court.png"
    )

    court_keypoints_per_frame = (
        tactical_view_converter.validate_keypoints(
            court_keypoints_per_frame
        )
    )

    tactical_player_positions = (
        tactical_view_converter.transform_players_to_tactical_view(
            court_keypoints_per_frame,
            player_tracks,
        )
    )

    speed_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters,
    )

    player_distances_per_frame = (
        speed_calculator.calculate_distance(
            tactical_player_positions
        )
    )

    player_speed_per_frame = (
        speed_calculator.calculate_speed(
            player_distances_per_frame,
            fps=fps,
        )
    )

    # Cumulative possession percentages are computed once.
    team_control_drawer = TeamBallControlDrawer()

    team_ball_control = (
        team_control_drawer.get_team_ball_control(
            player_assignment,
            ball_aquisition,
        )
    )

    # ---------------------------------------------------------
    # 4. Initialize drawing components.
    # ---------------------------------------------------------

    player_drawer = PlayerTracksDrawer()
    ball_drawer = BallTracksDrawer()
    court_drawer = CourtKeypointDrawer()
    frame_number_drawer = None
    pass_drawer = PassInterceptionDrawer()
    tactical_drawer = TacticalViewDrawer()
    speed_drawer = SpeedAndDistanceDrawer()

    # ---------------------------------------------------------
    # 5. Second pass: render chunks directly to the MP4 writer.
    # ---------------------------------------------------------

    print("Starting rendering pass...")

    frame_iterator = iter_video_frames(args.input_video)

    # Cumulative distance must persist across rendering chunks.
    cumulative_distances = {}

    try:
        with StreamingVideoWriter(
            args.output_video,
            fps,
            width,
            height,
        ) as writer:

            render_start = 0

            while True:
                frames = read_frame_batch(
                    frame_iterator,
                    args.batch_size,
                )

                if not frames:
                    break

                render_end = render_start + len(frames)

                print(
                    f"\rRendering frames "
                    f"{render_start + 1}-{render_end}",
                    end="",
                    flush=True,
                )

                # Slice only the compact results needed for this chunk.
                chunk_players = player_tracks[render_start:render_end]
                chunk_balls = ball_tracks[render_start:render_end]
                chunk_assignments = player_assignment[render_start:render_end]
                chunk_possession = ball_aquisition[render_start:render_end]
                chunk_keypoints = court_keypoints_per_frame[render_start:render_end]
                chunk_tactical = tactical_player_positions[render_start:render_end]
                chunk_distances = player_distances_per_frame[render_start:render_end]
                chunk_speeds = player_speed_per_frame[render_start:render_end]

                # Start with player overlays.
                output_frames = player_drawer.draw(
                    frames,
                    chunk_players,
                    chunk_assignments,
                    chunk_possession,
                )

                output_frames = ball_drawer.draw(
                    output_frames,
                    chunk_balls,
                )

                output_frames = court_drawer.draw(
                    output_frames,
                    chunk_keypoints,
                )

                # Preserve globally increasing frame numbers.
                for local_index, frame in enumerate(output_frames):
                    global_index = render_start + local_index

                    cv2.putText(
                        frame,
                        str(global_index),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    # Draw cumulative team possession on every frame.
                    team_control_drawer.draw_frame(
                        frame,
                        global_index,
                        team_ball_control,
                    )

                    # Draw cumulative pass/interception counts using
                    # the global frame index, not the chunk-local index.
                    pass_drawer.draw_frame(
                        frame,
                        global_index,
                        passes,
                        interceptions,
                    )

                # Keep cumulative distance consistent between chunks.
                chunk_distances_for_draw = [
                    dict(frame_distances)
                    for frame_distances in chunk_distances
                ]

                if chunk_distances_for_draw:
                    for player_id, total in cumulative_distances.items():
                        chunk_distances_for_draw[0][player_id] = (
                            chunk_distances_for_draw[0].get(player_id, 0)
                            + total
                        )

                output_frames = speed_drawer.draw(
                    output_frames,
                    chunk_players,
                    chunk_distances_for_draw,
                    chunk_speeds,
                )

                output_frames = tactical_drawer.draw(
                    output_frames,
                    tactical_view_converter.court_image_path,
                    tactical_view_converter.width,
                    tactical_view_converter.height,
                    tactical_view_converter.key_points,
                    chunk_tactical,
                    chunk_assignments,
                    chunk_possession,
                )

                # Update cumulative distances for the next chunk.
                for frame_distances in chunk_distances:
                    for player_id, distance in frame_distances.items():
                        cumulative_distances[player_id] = (
                            cumulative_distances.get(player_id, 0)
                            + distance
                        )

                # Write and release each chunk's output frames.
                for frame in output_frames:
                    writer.write(frame)

                render_start = render_end

                del frames
                del output_frames

    finally:
        frame_iterator.close()

    print(f"\nSaved output video: {args.output_video}")
    print(f"Processed {frame_number} frames successfully.")


if __name__ == "__main__":
    main()
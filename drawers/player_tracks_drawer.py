import cv2

from .utils import draw_traingle


class PlayerTracksDrawer:
    """
    A class responsible for drawing player bounding boxes,
    player IDs, confidence scores, and ball possession indicators.
    """

    def __init__(
        self,
        team_1_color=[255, 245, 238],
        team_2_color=[128, 0, 0]
    ):
        self.default_player_team_id = 1
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    def draw(
        self,
        video_frames,
        tracks,
        player_assignment,
        ball_aquisition
    ):
        """
        Draw player bounding boxes, IDs, confidence scores,
        and ball possession indicators.
        """

        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):

            frame = frame.copy()

            player_dict = tracks[frame_num]

            player_assignment_for_frame = player_assignment[frame_num]

            player_id_has_ball = ball_aquisition[frame_num]

            # Draw Players
            for track_id, player in player_dict.items():

                team_id = player_assignment_for_frame.get(
                    track_id,
                    self.default_player_team_id
                )

                if team_id == 1:
                    color = self.team_1_color
                else:
                    color = self.team_2_color

                # Player bounding box
                x1, y1, x2, y2 = map(
                    int,
                    player["bbox"]
                )

                frame = cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Get confidence
                confidence = player.get(
                    "confidence",
                    0.0
                )

                confidence_text = f"{confidence:.2f}"

                # Display Player ID + Confidence
                label = f"ID: {track_id} | Conf: {confidence_text}"

                label_y = max(
                    y1 - 8,
                    20
                )

                frame = cv2.putText(
                    frame,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                # Ball possession indicator
                if track_id == player_id_has_ball:

                    frame = draw_traingle(
                        frame,
                        player["bbox"],
                        (0, 0, 255)
                    )

            output_video_frames.append(frame)

        return output_video_frames
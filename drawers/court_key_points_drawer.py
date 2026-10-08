import supervision as sv


class CourtKeypointDrawer:

    def __init__(self):
        self.keypoint_color = '#ff2c2c'

    def draw(self, frames, court_keypoints):

        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            radius=8
        )

        vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1
        )

        output_frames = []

        for index, frame in enumerate(frames):
            annotated_frame = frame.copy()

            keypoints = court_keypoints[index]

            
            keypoints_numpy = keypoints.xy.cpu().numpy()

            key_points = sv.KeyPoints(
                xy=keypoints_numpy
            )

            
            annotated_frame = vertex_annotator.annotate(
                scene=annotated_frame,
                key_points=key_points
            )

            
            annotated_frame = vertex_label_annotator.annotate(
                scene=annotated_frame,
                key_points=key_points
            )

            output_frames.append(annotated_frame)

        return output_frames
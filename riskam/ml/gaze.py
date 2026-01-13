"""
ml.gaze

Gaze estimation for risk awareness.
"""

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
)

# Constants: yaw and pitch thresholds. The larger the threshold, the more lenient
YAW_THRESHOLD = 0
EYE_GAZE_THRESHOLD = 0

FACE_DETECTION_MODEL = "deepghs/yolo-face"


def detect_gaze(image: Image, visualize: bool = False) -> bool | None:
    """
    Detects the gaze of a person in the given image, if any.

    Returns True if the person is looking at the robot, False if not, None
    if no person is detected.
    """

    # Convert image to RGB
    image_np = np.array(image)
    h, w, _ = image_np.shape

    results = face_mesh.process(image_np)
    if not results.multi_face_landmarks:
        print("Mesh construction failed.")
        return None  # No face detected

    face_landmarks = results.multi_face_landmarks[0]

    # Ensure enough landmarks exist before accessing indices
    if len(face_landmarks.landmark) < 474:  # Ensure all eye & iris landmarks exist
        print("Incomplete face mesh data, skipping.")
        return None

    # Get key 2D facial landmarks
    left_eye = face_landmarks.landmark[33]  # Left eye corner
    right_eye = face_landmarks.landmark[263]  # Right eye corner
    nose_tip = face_landmarks.landmark[1]  # Nose tip
    chin = face_landmarks.landmark[199]  # Chin

    # Convert to image coordinates
    left_eye_pt = np.array([left_eye.x * w, left_eye.y * h])
    right_eye_pt = np.array([right_eye.x * w, right_eye.y * h])
    nose_tip_pt = np.array([nose_tip.x * w, nose_tip.y * h])
    chin_pt = np.array([chin.x * w, chin.y * h])

    # Compute yaw offset, normalized by face height
    face_height = np.linalg.norm(nose_tip_pt - chin_pt)
    eye_midpoint = (left_eye_pt + right_eye_pt) / 2
    yaw_offset = abs(nose_tip_pt[0] - eye_midpoint[0]) / face_height

    # Compute eye gaze direction
    left_iris = np.array(
        [face_landmarks.landmark[468].x * w, face_landmarks.landmark[468].y * h]
    )
    right_iris = np.array(
        [face_landmarks.landmark[473].x * w, face_landmarks.landmark[473].y * h]
    )

    left_iris_offset = abs(left_iris[0] - left_eye_pt[0]) / np.linalg.norm(
        right_eye_pt - left_eye_pt
    )
    right_iris_offset = abs(right_iris[0] - right_eye_pt[0]) / np.linalg.norm(
        right_eye_pt - left_eye_pt
    )

    eye_gaze_offset = (
        left_iris_offset + right_iris_offset
    ) / 2  # Average offset for both eyes

    print(f"Yaw Offset: {yaw_offset:.2f}, Eye Gaze Offset: {eye_gaze_offset:.2f}")

    is_aware = (np.abs(yaw_offset) < YAW_THRESHOLD) and (
        np.abs(eye_gaze_offset) < EYE_GAZE_THRESHOLD
    )
    return is_aware

    # # Convert image to RGB
    # image_np = np.array(image)

    # # # Detect and crop the face
    # # face_image = _detect_and_crop_face(image_np)

    # # if face_image is None:
    # #     print("No face detected.")
    # #     return None

    # results = face_mesh.process(image_np)

    # if not results.multi_face_landmarks:
    #     print("Mesh constructon failed.")
    #     return None  # No face detected

    # h, w, _ = image_np.shape  # Get image dimensions
    # face_landmarks = results.multi_face_landmarks[0]

    # # Get key 2D facial landmarks
    # left_eye = face_landmarks.landmark[33]  # Left eye corner
    # right_eye = face_landmarks.landmark[263]  # Right eye corner
    # nose_tip = face_landmarks.landmark[1]  # Nose tip

    # # Convert normalized landmark positions to image coordinates
    # left_eye_pt = np.array([left_eye.x * w, left_eye.y * h])
    # right_eye_pt = np.array([right_eye.x * w, right_eye.y * h])
    # nose_tip_pt = np.array([nose_tip.x * w, nose_tip.y * h])

    # eye_midpoint = (left_eye_pt + right_eye_pt) / 2

    # yaw_offset = abs(nose_tip_pt[0] - eye_midpoint[0]) / np.linalg.norm(
    #     right_eye_pt - left_eye_pt
    # )

    # print(f"Yaw Offset: {yaw_offset:.2f}")

    # is_aware = yaw_offset < YAW_THRESHOLD
    # return is_aware

    # # Get key facial landmarks
    # left_eye = face_landmarks.landmark[33]  # Left eye
    # right_eye = face_landmarks.landmark[263]  # Right eye
    # nose_base = face_landmarks.landmark[4]  # Nose base
    # chin = face_landmarks.landmark[152]  # Chin landmark

    # # Convert normalized positions to image coordinates
    # h, w, _ = face_image.shape  # Get image dimensions
    # left_eye_pt = (int(left_eye.x * w), int(left_eye.y * h))
    # right_eye_pt = (int(right_eye.x * w), int(right_eye.y * h))
    # nose_pt = (int(nose_base.x * w), int(nose_base.y * h))

    # # Compute midpoint between eyes
    # eye_mid_x = (left_eye.x + right_eye.x) / 2
    # eye_mid_y = (left_eye.y + right_eye.y) / 2
    # eye_mid_pt = (int(eye_mid_x * w), int(eye_mid_y * h))

    # # Compute total face height (eye midpoint to chin)
    # face_height = abs(eye_mid_y - chin.y) * 0.9  # slight scaling for stability

    # # Compute nose shift
    # eye_distance = (
    #     (right_eye.x - left_eye.x) ** 2 + (right_eye.y - left_eye.y) ** 2
    # ) ** 0.5  # Euclidean distance

    # nose_shift_x = abs(nose_base.x - eye_mid_x) / eye_distance  # Normalized yaw
    # nose_shift_y = abs(nose_base.y - eye_mid_y) / face_height  # Normalized pitch

    # print(f"Nose shift (x, y): {nose_shift_x:.2f}, {nose_shift_y:.2f}")

    # # Determine if the person is facing the camera
    # is_aware = nose_shift_x < YAW_THRESHOLD and nose_shift_y < PITCH_THRESHOLD

    # # Visualization: Draw eyes, nose, and gaze line
    # if visualize:
    #     cv2.circle(face_image, left_eye_pt, 3, (0, 255, 0), -1)  # Left eye (Green)
    #     cv2.circle(face_image, right_eye_pt, 3, (0, 255, 0), -1)  # Right eye (Green)
    #     cv2.circle(face_image, nose_pt, 3, (0, 0, 255), -1)  # Nose (Red)
    #     cv2.circle(face_image, eye_mid_pt, 3, (255, 0, 0), -1)  # Eye midpoint (Blue)
    #     cv2.line(
    #         face_image, eye_mid_pt, nose_pt, (255, 255, 0), 2
    #     )  # Line for yaw visualization

    #     # Display text with status
    #     status_text = "Aware" if is_aware else "Not Aware"
    #     color = (
    #         (0, 255, 0) if is_aware else (0, 0, 255)
    #     )  # Green for aware, Red for not aware
    #     cv2.putText(
    #         face_image,
    #         status_text,
    #         (nose_pt[0] - 40, nose_pt[1] - 20),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.5,
    #         color,
    #         2,
    #         cv2.LINE_AA,
    #     )

    #     cv2.imshow("Gaze Detection Debug", face_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # return is_aware


def _detect_and_crop_face(image_np: np.ndarray) -> np.ndarray | None:
    """Detects a face in a full-body image and returns a cropped face region."""
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # Convert to RGB
    results = face_detection.process(image_rgb)

    if not results.detections:
        return None  # No face found

    # Get bounding box of the first detected face
    h, w, _ = image_np.shape

    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    x, y, w_box, h_box = (
        int(bboxC.xmin * w),
        int(bboxC.ymin * h),
        int(bboxC.width * w),
        int(bboxC.height * h),
    )

    # Expand the bounding box a bit for better cropping
    padding = int(0.1 * w_box)
    x, y = max(x - padding, 0), max(y - padding, 0)
    w_box, h_box = min(w_box + 2 * padding, w - x), min(h_box + 2 * padding, h - y)

    # Crop face region
    face_crop = image_np[y : y + h_box, x : x + w_box]
    return face_crop


def _get_camera_matrix(image_width: int, image_height: int) -> np.ndarray:
    """Dynamically generates a camera matrix based on image dimensions."""
    focal_length = max(image_width, image_height) * 1.2
    camera_center = (image_width / 2, image_height / 2)

    return np.array(
        [
            [focal_length, 0, camera_center[0]],
            [0, focal_length, camera_center[1]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

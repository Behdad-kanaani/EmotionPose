import cv2
import mediapipe as mp
import math

# 1. Initialize MediaPipe modules
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Holistic model settings
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,   # Minimum detection confidence
    min_tracking_confidence=0.5,    # Minimum tracking confidence
    model_complexity=1              # Model complexity (0, 1, or 2)
)

# 2. Access the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit()

print("Webcam successfully opened. Press 'q' to quit.")

def get_landmark_coords(landmarks, idx, frame_shape):
    """
    Helper function to get landmark coordinates from a MediaPipe object.
    Returns None if the landmark does not exist.
    """
    if landmarks and idx < len(landmarks.landmark):
        x = int(landmarks.landmark[idx].x * frame_shape[1])
        y = int(landmarks.landmark[idx].y * frame_shape[0])
        return x, y
    return None, None

def calculate_distance(p1, p2):
    """
    Calculates the Euclidean distance between two points.
    """
    if p1 is None or p2 is None:
        return 0
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to receive frame. Is the webcam disconnected?")
        break

    # Convert image from BGR to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False # Make image non-writeable for better performance

    # Process the image with the Holistic model
    results = holistic.process(image)

    # Convert image back to writeable and BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # --- Draw body skeleton (Pose Landmarks) ---
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), # Landmark color
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)  # Connection color
    )

    emotion_text = "Neutral" # Default: Neutral

    # --- Emotion detection from face landmarks ---
    if results.face_landmarks:
        # Define important landmarks for emotion detection (based on MediaPipe Face Mesh)
        # Mouth corners
        p_mouth_left = get_landmark_coords(results.face_landmarks, 61, frame.shape)
        p_mouth_right = get_landmark_coords(results.face_landmarks, 291, frame.shape)
        # Upper and lower lip centers
        p_lip_upper_center = get_landmark_coords(results.face_landmarks, 0, frame.shape)
        p_lip_lower_center = get_landmark_coords(results.face_landmarks, 17, frame.shape)

        # Inner and outer eyebrows
        p_brow_inner_left = get_landmark_coords(results.face_landmarks, 55, frame.shape)
        p_brow_inner_right = get_landmark_coords(results.face_landmarks, 285, frame.shape)
        p_brow_outer_left = get_landmark_coords(results.face_landmarks, 105, frame.shape)
        p_brow_outer_right = get_landmark_coords(results.face_landmarks, 334, frame.shape)

        # Eye points (for detecting open/closed eyes)
        p_eye_left_upper = get_landmark_coords(results.face_landmarks, 159, frame.shape)
        p_eye_left_lower = get_landmark_coords(results.face_landmarks, 145, frame.shape)
        p_eye_right_upper = get_landmark_coords(results.face_landmarks, 386, frame.shape)
        p_eye_right_lower = get_landmark_coords(results.face_landmarks, 374, frame.shape)

        # Nose tip (for vertical reference)
        p_nose_tip = get_landmark_coords(results.face_landmarks, 1, frame.shape)

        # Ensure all necessary points exist before calculation
        if all([p_mouth_left, p_mouth_right, p_lip_upper_center, p_lip_lower_center,
                p_brow_inner_left, p_brow_inner_right, p_brow_outer_left, p_brow_outer_right,
                p_eye_left_upper, p_eye_left_lower, p_eye_right_upper, p_eye_right_lower, p_nose_tip]):

            # General metric for normalization with face size
            face_width = calculate_distance(get_landmark_coords(results.face_landmarks, 10, frame.shape),
                                            get_landmark_coords(results.face_landmarks, 339, frame.shape)) # Ear points
            if face_width == 0: # Prevent division by zero
                face_width = 1

            # --- Emotion detection criteria ---

            # 1. Happiness
            mouth_width = calculate_distance(p_mouth_left, p_mouth_right)
            mouth_height = calculate_distance(p_lip_upper_center, p_lip_lower_center)
            mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0

            # Check for upward movement of mouth corners (smile)
            # Compare Y of mouth corners with Y of nose tip
            mouth_corner_y_avg = (p_mouth_left[1] + p_mouth_right[1]) / 2
            if mouth_corner_y_avg < p_nose_tip[1] - (0.05 * frame.shape[0]) and mouth_aspect_ratio < 0.3: # Empirical threshold
                emotion_text = "Happiness (Smile)"

            # 2. Surprise
            # Raised eyebrows
            brow_left_y_avg = (p_brow_inner_left[1] + p_brow_outer_left[1]) / 2
            brow_right_y_avg = (p_brow_inner_right[1] + p_brow_outer_right[1]) / 2
            eye_left_y_avg = (p_eye_left_upper[1] + p_eye_left_lower[1]) / 2
            eye_right_y_avg = (p_eye_right_upper[1] + p_eye_right_lower[1]) / 2

            # Open mouth (jaw drop)
            jaw_drop_distance = calculate_distance(p_lip_upper_center, p_lip_lower_center) / face_width

            # Open eyes
            eye_left_openness = calculate_distance(p_eye_left_upper, p_eye_left_lower) / face_width
            eye_right_openness = calculate_distance(p_eye_right_upper, p_eye_right_lower) / face_width

            if (brow_left_y_avg < eye_left_y_avg - (0.04 * frame.shape[0]) and
                brow_right_y_avg < eye_right_y_avg - (0.04 * frame.shape[0]) and
                jaw_drop_distance > 0.08 and # Threshold for open mouth
                eye_left_openness > 0.04 and eye_right_openness > 0.04): # Threshold for open eyes
                emotion_text = "Surprise"

            # 3. Anger
            # Lowered and furrowed eyebrows
            # Compare Y of inner eyebrows with Y of nose tip (lower than usual)
            if (p_brow_inner_left[1] > p_nose_tip[1] + (0.01 * frame.shape[0]) and
                p_brow_inner_right[1] > p_nose_tip[1] + (0.01 * frame.shape[0]) and
                calculate_distance(p_brow_inner_left, p_brow_inner_right) < (0.08 * frame.shape[1]) and # Furrowed eyebrows
                mouth_aspect_ratio > 0.4): # Mouth might be slightly open and stretched
                emotion_text = "Anger (Frown)"

            # 4. Sadness
            # Downward movement of mouth corners
            if mouth_corner_y_avg > p_nose_tip[1] + (0.03 * frame.shape[0]) and mouth_aspect_ratio < 0.2: # Mouth slightly closed and corners down
                emotion_text = "Sadness"

    # Display the detected emotion on the image
    cv2.putText(image, f"Emotion: {emotion_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # --- Draw Face Mesh Landmarks and specific facial points ---
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION, # Draw general face mesh
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=0, circle_radius=1), # Landmark color
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1) # Connection color
        )

        # Example: Draw specific eyebrow points (as before)
        eyebrow_points = [
            105, 66, 107, 55, 65, 52, 53, 46,   # Right eyebrow
            334, 296, 283, 276, 285, 295, 282, 279 # Left eyebrow
        ]
        for idx in eyebrow_points:
            x, y = get_landmark_coords(results.face_landmarks, idx, frame.shape)
            if x is not None and y is not None:
                cv2.circle(image, (x, y), 2, (0, 255, 255), -1)

        # Example: Draw iris landmarks
        # The Holistic model returns iris points as part of face_landmarks.
        iris_landmarks_ids = [473, 474, 475, 476, 477,   # Right eye
                              468, 469, 470, 471, 472]    # Left eye
        for idx in iris_landmarks_ids:
            x, y = get_landmark_coords(results.face_landmarks, idx, frame.shape)
            if x is not None and y is not None:
                cv2.circle(image, (x, y), 3, (255, 0, 0), -1)

        # Example: Draw mouth landmarks
        mouth_outline_ids = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, # Upper lip
            308, 324, 318, 402, 317, 14, 87, 178, 88, 95 # Lower lip
        ]
        for idx in mouth_outline_ids:
            x, y = get_landmark_coords(results.face_landmarks, idx, frame.shape)
            if x is not None and y is not None:
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    # --- Draw Hand Mesh (Hand Landmarks) ---
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    cv2.imshow('Human Body and Face Landmark Detection (Press "q" to quit)', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

holistic.close()
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import copy
import time

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# Force resolution to 1280x720 30FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)

is_front_camera = False  # True for front/selfie, False for rear
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    hands_results = hands.process(rgb_frame)

    # Mirror frame if rear camera
    display_frame = cv2.flip(frame, 1) if not is_front_camera else frame.copy()

    def draw_landmarks_mirrored(landmarks, connections, is_mirror=True):
        """Draw landmarks and mirror x if needed."""
        to_draw = copy.deepcopy(landmarks) if is_mirror else landmarks
        if is_mirror:
            for lm in to_draw.landmark:
                lm.x = 1.0 - lm.x
        mp_drawing.draw_landmarks(display_frame, to_draw, connections)
        # Draw numbers
        for idx, lm in enumerate(landmarks.landmark):
            x = int(lm.x * w)
            y = int(lm.y * h)
            if is_mirror:
                x = w - x
            cv2.putText(display_frame, str(idx), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Pose landmarks
    if pose_results.pose_landmarks:
        draw_landmarks_mirrored(pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, not is_front_camera)
        landmarks = pose_results.pose_landmarks.landmark
        left_shoulder = (w - int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)) if not is_front_camera else \
                        (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        right_shoulder = (w - int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)) if not is_front_camera else \
                         (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
    else:
        left_shoulder = right_shoulder = None

    # Hands landmarks
    if hands_results.multi_hand_landmarks and left_shoulder and right_shoulder:
        for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            draw_landmarks_mirrored(hand_landmarks, mp_hands.HAND_CONNECTIONS, not is_front_camera)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_pos = (w - int(wrist.x * w), int(wrist.y * h)) if not is_front_camera else \
                        (int(wrist.x * w), int(wrist.y * h))
            label = hands_results.multi_handedness[hand_idx].classification[0].label
            if not is_front_camera:
                label = "Left" if label == "Right" else "Right"
            cv2.putText(display_frame, f"{label} Hand", (wrist_pos[0] - 30, wrist_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # --- Status texts ---
    # Bottom-left: camera status
    cam_status = "Press 'Space' to Adjust the Camera | 'Esc' to Quit"
    cv2.putText(display_frame, cam_status, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Bottom-right: resolution + FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    res_fps_text = f"{w}x{h} | {int(fps)} FPS"
    text_size, _ = cv2.getTextSize(res_fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_w, text_h = text_size
    cv2.putText(display_frame, res_fps_text, (w - text_w - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Pose + Hands Detection", display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == 32:  # SPACE key
        is_front_camera = not is_front_camera

cap.release()
cv2.destroyAllWindows()

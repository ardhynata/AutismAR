import cv2
import mediapipe as mp
import random
import time
import numpy as np

# --- Settings ---
ACTIVE_INTERVAL = 5   # object falling duration
REST_INTERVAL = 2     # feedback time
MAX_FAILS = 3

# --- Mediapipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# =============================
# Object drawing
# =============================
def draw_object(frame, obj):
    """Draw object as circle or PNG if available."""
    x, y, r = obj["x"], obj["y"], obj["r"]

    if obj.get("image") is None:
        # Draw circle
        cv2.circle(frame, (x, y), r, (0, 0, 255), -1)
    else:
        # Draw PNG sprite (future)
        h, w, _ = obj["image"].shape
        y1, y2 = y - h//2, y + h//2
        x1, x2 = x - w//2, x + w//2

        if 0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0]:
            overlay = frame[y1:y2, x1:x2]
            alpha = obj["image"][:, :, 3] / 255.0
            for c in range(3):
                overlay[:, :, c] = (alpha * obj["image"][:, :, c] +
                                    (1 - alpha) * overlay[:, :, c])
            frame[y1:y2, x1:x2] = overlay

    return frame

# =============================
# Mechanics
# =============================
def spawn_object(w):
    """Spawn new object at random x at top."""
    return {"x": random.randint(50, w - 50), "y": 50, "r": 30, "image": None}

def update_object(obj, speed, h):
    """Update object position. Return True if missed."""
    obj["y"] += speed
    return obj["y"] > h

def check_catch(results, obj, w, h, frame):
    """Check if hand is inside object."""
    if not results.pose_landmarks:
        return False

    landmarks = results.pose_landmarks.landmark
    hands = [mp_pose.PoseLandmark.LEFT_INDEX, mp_pose.PoseLandmark.RIGHT_INDEX]

    for hand in hands:
        px, py = int(landmarks[hand].x * w), int(landmarks[hand].y * h)
        cv2.circle(frame, (px, py), 10, (255, 0, 0), -1)  # draw hand

        dist = np.sqrt((px - obj["x"])**2 + (py - obj["y"])**2)
        if dist <= obj["r"] + 20:  # catch threshold
            return True
    return False

def update_phase(phase, elapsed, caught, score, fails, last_change, feedback, obj, w):
    """Game phase transitions."""
    speed = 10 + score // 3  # difficulty scaling

    if phase == "ACTIVE" and elapsed > ACTIVE_INTERVAL:
        feedback = "Missed!"
        fails += 1
        phase = "REST"
        last_change = time.time()

    elif phase == "REST" and elapsed > REST_INTERVAL:
        obj = spawn_object(w)
        phase = "ACTIVE"
        last_change = time.time()
        feedback = ""

    return phase, score, fails, last_change, feedback, obj, speed

# =============================
# UI
# =============================
def draw_ui(frame, phase, score, fails, feedback, remaining, w, h, game_over):
    cv2.putText(frame, f"Score: {score}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"Fails: {fails}/{MAX_FAILS}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

    if game_over:
        cv2.putText(frame, "GAME OVER", (w // 2 - 200, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 8, cv2.LINE_AA)
        cv2.putText(frame, "Press R to restart", (w // 2 - 180, h // 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        return

    if phase == "ACTIVE":
        cv2.putText(frame, str(remaining), (w // 2 - 50, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 4, (0, 255, 255), 6, cv2.LINE_AA)
    elif phase == "REST":
        cv2.putText(frame, feedback, (w // 2 - 150, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 3,
                    (0, 255, 0) if feedback == "Nice!" else (0, 0, 255), 6, cv2.LINE_AA)

def show_start_screen(cap):
    """Start screen before SPACE is pressed."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (w - 50, h - 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.putText(frame, "CATCH IT!", (w // 2 - 250, h // 3),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.putText(frame, "Use your HANDS to catch circles", (w // 2 - 250, h // 2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(frame, "Press SPACE to start", (w // 2 - 180, h // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, "Press Q or ESC to quit", (w // 2 - 180, h // 2 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        cv2.imshow("Catch the Object", frame)
        key = cv2.waitKey(5) & 0xFF
        if key in [27, ord('q')]:
            return False
        if key == 32:  # SPACE
            return True
    return False

# =============================
# Main Loop
# =============================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)

    while True:  # restart loop
        if not show_start_screen(cap):
            cap.release()
            cv2.destroyAllWindows()
            return

        # Game init
        phase = "ACTIVE"
        last_change = time.time()
        score = 0
        fails = 0
        feedback = ""
        obj = None
        speed = 10
        game_over = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if obj is None:
                obj = spawn_object(w)

            active_time = ACTIVE_INTERVAL
            elapsed = time.time() - last_change
            remaining = (active_time if phase == "ACTIVE" else REST_INTERVAL) - int(elapsed)

            # update game state
            if not game_over:
                if phase == "ACTIVE":
                    # check catch instantly
                    if check_catch(results, obj, w, h, frame):
                        score += 1
                        feedback = "Nice!"
                        phase = "REST"
                        last_change = time.time()
                    else:
                        missed = update_object(obj, speed, h)
                        if missed:
                            fails += 1
                            feedback = "Missed!"
                            phase = "REST"
                            last_change = time.time()

                # handle normal transition (only for missed)
                phase, score, fails, last_change, feedback, obj, speed = update_phase(
                    phase, elapsed, False, score, fails, last_change, feedback, obj, w
                )

            # draw object
            if obj is not None and phase == "ACTIVE":
                frame = draw_object(frame, obj)

            # check game over
            if fails >= MAX_FAILS:
                game_over = True

            # draw UI
            draw_ui(frame, phase, score, fails, feedback, remaining, w, h, game_over)

            cv2.imshow("Catch the Object", frame)
            key = cv2.waitKey(5) & 0xFF
            if key in [27, ord('q')]:
                cap.release()
                cv2.destroyAllWindows()
                return
            if game_over and key == ord('r'):
                break

if __name__ == "__main__":
    main()

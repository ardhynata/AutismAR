import cv2
import mediapipe as mp
import random
import time
import numpy as np

# --- Settings ---
GRID_SIZE = 3
ACTIVE_INTERVAL = 7   # initial safe cell active time
REST_INTERVAL = 3     # feedback time
MAX_FAILS = 3

# --- Mediapipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# =============================
# Utility Functions
# =============================
def get_trapezoid_points(w, h):
    """Return trapezoid floor corners."""
    floor_top = int(h * 0.5)
    bottom_margin = int(h * 0.07)   # raise bottom by 7% of frame height
    top_offset = int(w * 0.05)      # widen top by 5%

    top_left = (int(w * 0.25) - top_offset, floor_top)
    top_right = (int(w * 0.75) + top_offset, floor_top)
    bottom_left = (0, h - bottom_margin)
    bottom_right = (w, h - bottom_margin)

    return top_left, top_right, bottom_left, bottom_right


def draw_grid(frame, safe_cell, trapezoid):
    """Draw trapezoid grid with safe cell highlighted."""
    top_left, top_right, bottom_left, bottom_right = trapezoid
    overlay = frame.copy()

    for gy in range(GRID_SIZE):
        y_t, y_b = gy / GRID_SIZE, (gy + 1) / GRID_SIZE

        left_top = (int(top_left[0] + (bottom_left[0]-top_left[0])*y_t),
                    int(top_left[1] + (bottom_left[1]-top_left[1])*y_t))
        right_top = (int(top_right[0] + (bottom_right[0]-top_right[0])*y_t),
                     int(top_right[1] + (bottom_right[1]-top_right[1])*y_t))
        left_bottom = (int(top_left[0] + (bottom_left[0]-top_left[0])*y_b),
                       int(top_left[1] + (bottom_left[1]-top_left[1])*y_b))
        right_bottom = (int(top_right[0] + (bottom_right[0]-top_right[0])*y_b),
                        int(top_right[1] + (bottom_right[1]-top_right[1])*y_b))

        for gx in range(GRID_SIZE):
            x_t, x_b = gx / GRID_SIZE, (gx + 1) / GRID_SIZE

            cell_tl = (int(left_top[0] + (right_top[0]-left_top[0])*x_t),
                       int(left_top[1] + (right_top[1]-left_top[1])*x_t))
            cell_tr = (int(left_top[0] + (right_top[0]-left_top[0])*x_b),
                       int(left_top[1] + (right_top[1]-left_top[1])*x_b))
            cell_bl = (int(left_bottom[0] + (right_bottom[0]-left_bottom[0])*x_t),
                       int(left_bottom[1] + (right_bottom[1]-left_bottom[1])*x_t))
            cell_br = (int(left_bottom[0] + (right_bottom[0]-left_bottom[0])*x_b),
                       int(left_bottom[1] + (right_bottom[1]-left_bottom[1])*x_b))

            pts = np.array([cell_tl, cell_tr, cell_br, cell_bl], dtype=np.int32)

            if safe_cell is not None and (gx, gy) == safe_cell:
                cv2.fillPoly(overlay, [pts], (0, 200, 0))   # safe = green
                cv2.polylines(overlay, [pts], True, (0, 255, 255), 4)  # highlight border
            else:
                cv2.fillPoly(overlay, [pts], (40, 40, 40))  # unsafe = dark gray

    return cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)


def check_feet_in_safe_cell(results, w, h, safe_cell, trapezoid, frame):
    """Check if feet are inside safe cell polygon."""
    if safe_cell is None or not results.pose_landmarks:
        return False

    landmarks = results.pose_landmarks.landmark
    feet = [mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

    top_left, top_right, bottom_left, bottom_right = trapezoid
    gy, gx = safe_cell[1], safe_cell[0]

    # compute safe cell polygon
    y_t, y_b = gy / GRID_SIZE, (gy + 1) / GRID_SIZE
    left_top = (int(top_left[0] + (bottom_left[0]-top_left[0])*y_t),
                int(top_left[1] + (bottom_left[1]-top_left[1])*y_t))
    right_top = (int(top_right[0] + (bottom_right[0]-top_right[0])*y_t),
                 int(top_right[1] + (bottom_right[1]-top_right[1])*y_t))
    left_bottom = (int(top_left[0] + (bottom_left[0]-top_left[0])*y_b),
                   int(top_left[1] + (bottom_left[1]-top_left[1])*y_b))
    right_bottom = (int(top_right[0] + (bottom_right[0]-top_right[0])*y_b),
                    int(top_right[1] + (bottom_right[1]-top_right[1])*y_b))

    x_t, x_b = gx / GRID_SIZE, (gx + 1) / GRID_SIZE
    cell_tl = (int(left_top[0] + (right_top[0]-left_top[0])*x_t),
               int(left_top[1] + (right_top[1]-left_top[1])*x_t))
    cell_tr = (int(left_top[0] + (right_top[0]-left_top[0])*x_b),
               int(left_top[1] + (right_top[1]-left_top[1])*x_b))
    cell_bl = (int(left_bottom[0] + (right_bottom[0]-left_bottom[0])*x_t),
               int(left_bottom[1] + (right_bottom[1]-left_bottom[1])*x_t))
    cell_br = (int(left_bottom[0] + (right_bottom[0]-left_bottom[0])*x_b),
               int(left_bottom[1] + (right_bottom[1]-left_bottom[1])*x_b))
    pts = np.array([cell_tl, cell_tr, cell_br, cell_bl], dtype=np.int32)

    inside_any = False
    for foot in feet:
        px, py = int(landmarks[foot].x * w), int(landmarks[foot].y * h)
        inside = cv2.pointPolygonTest(pts, (px, py), False) >= 0
        color = (0, 255, 0) if inside else (0, 0, 255)
        cv2.circle(frame, (px, py), 10, color, -1)
        if inside:
            inside_any = True

    return inside_any


def update_phase(phase, elapsed, was_on_safe, score, fails, last_change, feedback, prev_cell):
    """Update phase transitions, difficulty scaling, and scoring logic."""
    safe_cell = None
    active_time = max(3, ACTIVE_INTERVAL - score // 3)  # shrink active interval every 3 points

    if phase == "ACTIVE" and elapsed > active_time:
        feedback = "Nice!" if was_on_safe else "Try Again"
        if was_on_safe:
            score += 1
        else:
            fails += 1
        phase = "REST"
        last_change = time.time()

    elif phase == "REST" and elapsed > REST_INTERVAL:
        # pick a new safe cell (not same as before)
        while True:
            new_cell = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if new_cell != prev_cell:
                safe_cell = new_cell
                break
        phase = "ACTIVE"
        last_change = time.time()
        feedback = ""   # clear feedback

    return phase, score, fails, last_change, feedback, safe_cell


def draw_ui(frame, phase, score, fails, feedback, remaining, w, h, game_over):
    """Draw score, countdown, feedback, and Game Over text."""
    cv2.putText(frame, f"Skor: {score}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"Gagal: {fails}/{MAX_FAILS}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

    if game_over:
        cv2.putText(frame, "GAME OVER", (w // 2 - 250, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 8, cv2.LINE_AA)
        cv2.putText(frame, "Tekan R untuk bermain lagi", (w // 2 - 220, h // 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)   # ADDED
        return

    if phase == "ACTIVE":
        color = (0, 255, 255)  # yellow
        if remaining <= 3 and int(time.time() * 2) % 2 == 0:
            color = (0, 0, 255)  # flashing red
        cv2.putText(frame, str(remaining), (w // 2 - 70, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 5, color, 8, cv2.LINE_AA)

    elif phase == "REST":
        cv2.putText(frame, feedback, (w // 2 - 200, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 3,
                    (0, 255, 0) if feedback == "Nice!" else (0, 0, 255), 6, cv2.LINE_AA)


def show_start_screen(cap):
    """Display start screen until SPACE is pressed."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (w - 50, h - 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.putText(frame, "FLOOR IS LAVA!", (w // 2 - 250, h // 3),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.putText(frame, "Lompatlah ke kotak HIJAU", (w // 2 - 200, h // 2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(frame, "Tekan SPASI untuk mulai", (w // 2 - 200, h // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, "Tekan Q atau ESC untuk keluar", (w // 2 - 230, h // 2 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        cv2.imshow("Floor is Lava", frame)
        key = cv2.waitKey(5) & 0xFF
        if key in [27, ord('q')]:  # ESC or q
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

    while True:   # ADDED for restart loop
        # Show start screen
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
        safe_cell = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        prev_cell = safe_cell
        was_on_safe = False
        game_over = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # flip horizontally
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            trapezoid = get_trapezoid_points(w, h)
            active_time = max(3, ACTIVE_INTERVAL - score // 3)
            elapsed = time.time() - last_change
            remaining = (active_time if phase == "ACTIVE" else REST_INTERVAL) - int(elapsed)

            # update phase & scoring
            if not game_over:
                phase, score, fails, last_change, feedback, new_safe_cell = update_phase(
                    phase, elapsed, was_on_safe, score, fails, last_change, feedback, prev_cell
                )
                if new_safe_cell is not None:
                    prev_cell = safe_cell
                    safe_cell = new_safe_cell

            # draw grid
            frame = draw_grid(frame, safe_cell, trapezoid)

            # check if player is on safe cell
            standing_safe = check_feet_in_safe_cell(results, w, h, safe_cell, trapezoid, frame)
            was_on_safe = standing_safe if phase == "ACTIVE" else False

            # check game over
            if fails >= MAX_FAILS:
                game_over = True

            # draw UI
            draw_ui(frame, phase, score, fails, feedback, remaining, w, h, game_over)

            cv2.imshow("Floor is Lava", frame)
            key = cv2.waitKey(5) & 0xFF
            if key in [27, ord('q')]:  # quit
                cap.release()
                cv2.destroyAllWindows()
                return
            if game_over and key == ord('r'):  # restart
                break  # break inner game loop → restart outer while loop

if __name__ == "__main__":
    main()

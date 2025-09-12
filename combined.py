import cv2
import mediapipe as mp
import random
import time
import numpy as np

# --- Settings ---
BALL_SPEED = 4
GRID_SIZE = 3
ACTIVE_INTERVAL = 7   # active safe-cell duration (seconds)
REST_INTERVAL = 3     # rest/feedback duration (seconds)
MAX_FAILS = 3

# --- Mediapipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# =============================
# Utility / Geometry
# =============================
def get_trapezoid_points(w, h):
    """Return trapezoid floor corners."""
    floor_top = int(h * 0.5)
    bottom_margin = int(h * 0.07)
    top_offset = int(w * 0.05)

    top_left = (int(w * 0.25) - top_offset, floor_top)
    top_right = (int(w * 0.75) + top_offset, floor_top)
    bottom_left = (0, h - bottom_margin)
    bottom_right = (w, h - bottom_margin)

    return top_left, top_right, bottom_left, bottom_right


def cell_polygon_for(gx, gy, trapezoid):
    """Return polygon points for cell (gx, gy)."""
    top_left, top_right, bottom_left, bottom_right = trapezoid
    y_t, y_b = gy / GRID_SIZE, (gy + 1) / GRID_SIZE
    left_top = (int(top_left[0] + (bottom_left[0] - top_left[0]) * y_t),
                int(top_left[1] + (bottom_left[1] - top_left[1]) * y_t))
    right_top = (int(top_right[0] + (bottom_right[0] - top_right[0]) * y_t),
                 int(top_right[1] + (bottom_right[1] - top_right[1]) * y_t))
    left_bottom = (int(top_left[0] + (bottom_left[0] - top_left[0]) * y_b),
                   int(top_left[1] + (bottom_left[1] - top_left[1]) * y_b))
    right_bottom = (int(top_right[0] + (bottom_right[0] - top_right[0]) * y_b),
                    int(top_right[1] + (bottom_right[1] - top_right[1]) * y_b))

    x_t, x_b = gx / GRID_SIZE, (gx + 1) / GRID_SIZE
    cell_tl = (int(left_top[0] + (right_top[0] - left_top[0]) * x_t),
               int(left_top[1] + (right_top[1] - left_top[1]) * x_t))
    cell_tr = (int(left_top[0] + (right_top[0] - left_top[0]) * x_b),
               int(left_top[1] + (right_top[1] - left_top[1]) * x_b))
    cell_bl = (int(left_bottom[0] + (right_bottom[0] - left_bottom[0]) * x_t),
               int(left_bottom[1] + (right_bottom[1] - left_bottom[1]) * x_t))
    cell_br = (int(left_bottom[0] + (right_bottom[0] - left_bottom[0]) * x_b),
               int(left_bottom[1] + (right_bottom[1] - left_bottom[1]) * x_b))

    pts = np.array([cell_tl, cell_tr, cell_br, cell_bl], dtype=np.int32)
    return pts


def cell_center_x_for(gx, gy, trapezoid):
    """Return center x coordinate of a cell (useful for spawning ball)."""
    pts = cell_polygon_for(gx, gy, trapezoid)
    xs = pts[:, 0]
    return int(xs.mean())


# =============================
# Grid Drawing & Feet check
# =============================
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
                cv2.polylines(overlay, [pts], True, (0, 255, 255), 4)  # border
            else:
                cv2.fillPoly(overlay, [pts], (40, 40, 40))  # unsafe = dark gray

    return cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)


def check_feet_in_safe_cell(results, w, h, safe_cell, trapezoid, frame):
    """Check if feet are inside safe cell polygon."""
    if safe_cell is None or not results.pose_landmarks:
        return False

    landmarks = results.pose_landmarks.landmark
    feet = [mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

    gy, gx = safe_cell[1], safe_cell[0]
    pts = cell_polygon_for(gx, gy, trapezoid)

    inside_any = False
    for foot in feet:
        px, py = int(landmarks[foot].x * w), int(landmarks[foot].y * h)
        inside = cv2.pointPolygonTest(pts, (px, py), False) >= 0
        color = (0, 255, 0) if inside else (0, 0, 255)
        cv2.circle(frame, (px, py), 10, color, -1)
        if inside:
            inside_any = True

    return inside_any


# =============================
# Object (Ball) mechanics
# =============================
def spawn_object_for_safe(gx_safe, trapezoid, w):
    if gx_safe == 0:
        allowed = [0, 1]
    elif gx_safe == 2:
        allowed = [1, 2]
    else:
        allowed = [0, 1, 2]

    chosen_col = random.choice(allowed)
    return chosen_col


def build_object_from_col(col, gy_active, trapezoid, frame_h):
    pts = cell_polygon_for(col, gy_active, trapezoid)
    top_left, top_right = pts[0], pts[1]
    x_min, x_max = top_left[0], top_right[0]
    x_spawn = random.randint(min(x_min, x_max), max(x_min, x_max))
    y_spawn = 0
    obj = {"x": x_spawn, "y": y_spawn, "r": 30}
    return obj


def update_object_pos(obj, speed, h):
    obj["y"] += speed
    return obj["y"] > h - 60


def draw_object(frame, obj):
    x, y, r = obj["x"], int(obj["y"]), obj["r"]
    cv2.circle(frame, (x, y), r, (0, 0, 255), -1)
    cv2.ellipse(frame, (x, min(frame.shape[0] - 30, y + r + 10)),
                (r, int(r / 2)), 0, 0, 360, (50, 50, 50), -1)
    return frame


def check_catch(results, obj, w, h, frame):
    if not results.pose_landmarks:
        return False

    landmarks = results.pose_landmarks.landmark
    hands = [mp_pose.PoseLandmark.LEFT_INDEX, mp_pose.PoseLandmark.RIGHT_INDEX]

    for hand in hands:
        px, py = int(landmarks[hand].x * w), int(landmarks[hand].y * h)
        cv2.circle(frame, (px, py), 10, (255, 0, 0), -1)
        dist = np.sqrt((px - obj["x"])**2 + (py - obj["y"])**2)
        if dist <= obj["r"] + 25:
            return True
    return False


# =============================
# Phases & UI
# =============================
def update_phase_floor(phase, elapsed, was_on_safe, score, fails, last_change, feedback, prev_cell):
    safe_cell = None
    active_time = max(3, ACTIVE_INTERVAL - score // 3)

    if phase == "ACTIVE" and elapsed > active_time:
        if was_on_safe:
            feedback = "Nice (Floor)!"
            score += 1
        else:
            feedback = "Try Again (Floor)"
            fails += 1
        phase = "REST"
        last_change = time.time()

    elif phase == "REST" and elapsed > REST_INTERVAL:
        while True:
            new_cell = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if new_cell != prev_cell:
                safe_cell = new_cell
                break
        phase = "ACTIVE"
        last_change = time.time()
        feedback = ""

    return phase, score, fails, last_change, feedback, safe_cell


def draw_ui(frame, phase, score, fails, feedback, remaining, w, h, game_over):
    cv2.putText(frame, f"Score: {score}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"Fails: {fails}/{MAX_FAILS}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

    if game_over:
        cv2.putText(frame, "GAME OVER", (w // 2 - 220, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 8, cv2.LINE_AA)
        cv2.putText(frame, "Press R to restart", (w // 2 - 200, h // 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        return

    if phase == "ACTIVE":
        color = (0, 255, 255)
        if remaining <= 3 and int(time.time() * 2) % 2 == 0:
            color = (0, 0, 255)
        cv2.putText(frame, str(remaining), (w // 2 - 70, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 5, color, 8, cv2.LINE_AA)
    elif phase == "REST":
        cv2.putText(frame, feedback, (w // 2 - 200, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0,
                    (0, 255, 0) if "Nice" in feedback else (0, 0, 255), 4, cv2.LINE_AA)


def show_start_screen(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (w - 50, h - 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.putText(frame, "FLOOR + CATCH (Combined)", (w // 2 - 360, h // 3),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, "Stand on the GREEN cell (feet) AND catch the BALL (hands)", (w // 2 - 520, h // 2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Space: Start  |  Q/ESC: Quit", (w // 2 - 220, h // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Lava + Catch", frame)
        key = cv2.waitKey(5) & 0xFF
        if key in [27, ord('q')]:
            return False
        if key == 32:
            return True
    return False


# =============================
# Main combined loop
# =============================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)

    while True:
        if not show_start_screen(cap):
            cap.release()
            cv2.destroyAllWindows()
            return

        phase = "ACTIVE"
        last_change = time.time()
        score = 0
        fails = 0
        feedback = ""
        safe_cell = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        prev_cell = safe_cell
        was_on_safe = False
        game_over = False

        obj = None
        speed = BALL_SPEED

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            trapezoid = get_trapezoid_points(w, h)
            active_time = max(3, ACTIVE_INTERVAL - score // 3)
            elapsed = time.time() - last_change
            remaining = (active_time if phase == "ACTIVE" else REST_INTERVAL) - int(elapsed)

            frame = draw_grid(frame, safe_cell, trapezoid)
            standing_safe = check_feet_in_safe_cell(results, w, h, safe_cell, trapezoid, frame)
            was_on_safe = standing_safe if phase == "ACTIVE" else False

            # Spawn object if needed
            if obj is None and phase == "ACTIVE":
                chosen_col = spawn_object_for_safe(safe_cell[0], trapezoid, w)
                obj = build_object_from_col(chosen_col, safe_cell[1], trapezoid, h)
                obj["x"] += random.randint(-20, 20)
                speed = BALL_SPEED + score // 3

            # Ball mechanics
            if not game_over and phase == "ACTIVE" and obj is not None:
                if check_catch(results, obj, w, h, frame):
                    if standing_safe:
                        score += 1
                        feedback = "Nice (Catch)!"
                    else:
                        fails += 1
                        feedback = "Must stand on safe cell!"
                    phase = "REST"
                    last_change = time.time()
                    obj = None
                else:
                    missed = update_object_pos(obj, speed, h)
                    if missed:
                        fails += 1
                        feedback = "Missed the ball!"
                        phase = "REST"
                        last_change = time.time()
                        obj = None

            # Floor phase handling
            if not game_over:
                prev_phase = phase
                phase, score, fails, last_change, feedback_floor, new_safe_cell = update_phase_floor(
                    phase, elapsed, was_on_safe, score, fails, last_change, feedback, prev_cell
                )
                if new_safe_cell is not None:
                    # <<< NEW >>> If ball still exists when safe cell changes, count as fail
                    if obj is not None:
                        fails += 1
                        feedback = "Ball dropped!"
                        obj = None
                        phase = "REST"
                        last_change = time.time()
                    prev_cell = safe_cell
                    safe_cell = new_safe_cell
                if feedback_floor:
                    feedback = feedback_floor

            if obj is not None and phase == "ACTIVE":
                frame = draw_object(frame, obj)

            if fails >= MAX_FAILS:
                game_over = True

            draw_ui(frame, phase, score, fails, feedback, remaining, w, h, game_over)

            cv2.imshow("Lava + Catch", frame)
            key = cv2.waitKey(5) & 0xFF
            if key in [27, ord('q')]:
                cap.release()
                cv2.destroyAllWindows()
                return
            if game_over and key == ord('r'):
                break

if __name__ == "__main__":
    main()

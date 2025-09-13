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
MAX_GAMES = 10        # maximum safe zone changes

# --- Mediapipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# =============================
# Utility / Geometry
# =============================
def get_trapezoid_points(w, h):
    floor_top = int(h * 0.5)
    bottom_margin = int(h * 0.07)
    top_offset = int(w * 0.05)
    top_left = (int(w * 0.25) - top_offset, floor_top)
    top_right = (int(w * 0.75) + top_offset, floor_top)
    bottom_left = (0, h - bottom_margin)
    bottom_right = (w, h - bottom_margin)
    return top_left, top_right, bottom_left, bottom_right

def cell_polygon_for(gx, gy, trapezoid):
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

# =============================
# Grid & feet check
# =============================
def draw_grid(frame, safe_cell, trapezoid):
    overlay = frame.copy()
    top_left, top_right, bottom_left, bottom_right = trapezoid
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
                cv2.fillPoly(overlay, [pts], (0, 200, 0))
                cv2.polylines(overlay, [pts], True, (0, 255, 255), 4)
            else:
                cv2.fillPoly(overlay, [pts], (40, 40, 40))
    return cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

def check_feet_in_safe_cell(results, w, h, safe_cell, trapezoid, frame):
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
        cv2.circle(frame, (px, py), 10, (0, 255, 0) if inside else (0, 0, 255), -1)
        if inside:
            inside_any = True
    return inside_any

# =============================
# Ball mechanics
# =============================
def spawn_object_for_safe(gx_safe, trapezoid, w):
    allowed = [0, 1] if gx_safe == 0 else [1, 2] if gx_safe == 2 else [0, 1, 2]
    return random.choice(allowed)

def build_object_from_col(col, gy_active, trapezoid, frame_h):
    pts = cell_polygon_for(col, gy_active, trapezoid)
    x_min, x_max = pts[0][0], pts[1][0]
    return {"x": random.randint(min(x_min, x_max), max(x_min, x_max)), "y": 0, "r": 30}

def update_object_pos(obj, speed, h):
    obj["y"] += speed
    return obj["y"] > h - 60

def draw_object(frame, obj):
    x, y, r = obj["x"], int(obj["y"]), obj["r"]
    cv2.circle(frame, (x, y), r, (0, 0, 255), -1)
    cv2.ellipse(frame, (x, min(frame.shape[0]-30, y+r+10)), (r, int(r/2)), 0, 0, 360, (50,50,50), -1)
    return frame

def check_catch(results, obj, w, h, frame):
    if not results.pose_landmarks:
        return False
    landmarks = results.pose_landmarks.landmark
    hands = [mp_pose.PoseLandmark.LEFT_INDEX, mp_pose.PoseLandmark.RIGHT_INDEX]
    for hand in hands:
        px, py = int(landmarks[hand].x * w), int(landmarks[hand].y * h)
        cv2.circle(frame, (px, py), 10, (255, 0, 0), -1)
        if np.sqrt((px - obj["x"])**2 + (py - obj["y"])**2) <= obj["r"] + 25:
            return True
    return False

# =============================
# Main loop
# =============================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)

    balls_caught_total = 0
    safe_zone_total = 0
    safe_zone_changes = 0
    balls_caught = 0
    safe_zone_count = 0
    safe_cell = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    obj = None
    last_change = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        trapezoid = get_trapezoid_points(w, h)
        elapsed = time.time() - last_change

        frame = draw_grid(frame, safe_cell, trapezoid)
        if check_feet_in_safe_cell(results, w, h, safe_cell, trapezoid, frame):
            safe_zone_count += 1

        # Spawn ball
        if obj is None:
            chosen_col = spawn_object_for_safe(safe_cell[0], trapezoid, w)
            obj = build_object_from_col(chosen_col, safe_cell[1], trapezoid, h)

        # Ball mechanics
        if obj is not None:
            if check_catch(results, obj, w, h, frame):
                balls_caught += 1
                obj = None
            else:
                missed = update_object_pos(obj, BALL_SPEED, h)
                if missed:
                    obj = None

        if obj is not None:
            frame = draw_object(frame, obj)

        # Safe zone change = one "game"
        if elapsed > ACTIVE_INTERVAL:
            safe_zone_changes += 1
            balls_caught_total += balls_caught
            safe_zone_total += safe_zone_count
            balls_caught = 0
            safe_zone_count = 0
            safe_cell = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            last_change = time.time()
            obj = None
            if safe_zone_changes >= MAX_GAMES:
                break

        cv2.imshow("Floor + Catch", frame)
        key = cv2.waitKey(5) & 0xFF
        if key in [27, ord('q')]:  # ESC or Q to quit anytime
            break

    # Final summary
    print(f"\n--- GAME SUMMARY ---")
    print(f"Balls caught: {balls_caught_total}/{MAX_GAMES}")
    print(f"Safe zone standing: {safe_zone_total}/{MAX_GAMES}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

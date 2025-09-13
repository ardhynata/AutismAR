import cv2
import mediapipe as mp
import random
import time
import numpy as np
from utils.game import ExitProgram

# --- Settings ---
GRID_SIZE = 3
ACTIVE_INTERVAL = 2
REST_INTERVAL = 3
MAX_GAMES = 2

MULTICOLOR_MODE = True  # True → 3 squares with colors, False → single green square
REQUIRE_BOTH_FEET = True  # set to False to allow just one foot
FIREBALL_MODE = True         # enable/disable fireball challenge
FIREBALL_HOLD_TIME = 3       # seconds required to hold fireball

# --- Colors ---
COLORS = {
    "RED": (0, 0, 255),
    "GREEN": (0, 200, 0),
    "BLUE": (255, 0, 0)
}

# --- Mediapipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# =============================
# Shared Scoring System
# =============================
class GameSession:
    def __init__(self, max_games, game_name="Game"):
        self.max_games = max_games
        self.game_name = game_name
        self.current_round = 0
        self.results = []

    def record_result(self, success: bool):
        if self.current_round < self.max_games:
            self.results.append("Success" if success else "Fail")
            self.current_round += 1

    def is_finished(self):
        return self.current_round >= self.max_games

    def summary(self):
        success_count = self.results.count("Success")
        fail_count = self.results.count("Fail")
        return success_count, fail_count, self.results

# =============================
# Utility Functions
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

def draw_grid(frame, active_cells, safe_color_name, trapezoid, multicolor):
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

            if multicolor:
                if (gx, gy) in active_cells:
                    color = active_cells[(gx, gy)]
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.polylines(overlay, [pts], True, (255, 255, 255), 3)
                else:
                    cv2.fillPoly(overlay, [pts], (40, 40, 40))
            else:  # single green square
                if active_cells is not None and (gx, gy) == active_cells:
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
    top_left, top_right, bottom_left, bottom_right = trapezoid
    gy, gx = safe_cell[1], safe_cell[0]
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
    
    inside_count = 0
    for foot in feet:
        px, py = int(landmarks[foot].x * w), int(landmarks[foot].y * h)
        inside = cv2.pointPolygonTest(pts, (px, py), False) >= 0
        color = (0, 255, 0) if inside else (0, 0, 255)
        cv2.circle(frame, (px, py), 10, color, -1)
        if inside:
            inside_count += 1

    if REQUIRE_BOTH_FEET:
        return inside_count == len(feet)
    else:
        return inside_count > 0


def draw_ui(frame, phase, score, feedback, remaining, w, h, game_count, max_games, multicolor, safe_color_name):
    cv2.putText(frame, f"Safe Score: {score}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Game: {game_count}/{max_games}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if phase == "ACTIVE":
        color = (0, 255, 255)
        if remaining <= 3 and int(time.time() * 2) % 2 == 0:
            color = (0, 0, 255)
        cv2.putText(frame, str(remaining), (w // 2 - 70, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 5, color, 8, cv2.LINE_AA)
        if multicolor:
            cv2.putText(frame, f"SAFE: {safe_color_name}", (w//2 - 200, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
    elif phase == "REST":
        cv2.putText(frame, feedback, (w // 2 - 200, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 3,
                    (0, 255, 0) if feedback == "Nice!" else (0, 0, 255), 6, cv2.LINE_AA)

# =============================
# Main Run Function
# =============================
def run(camera_stream, display_manager, config):
    window_name = config.WINDOW_NAME
    screen_width, screen_height = display_manager.get_screen_size()
    window_width, window_height = display_manager.compute_window_size(
        screen_width, screen_height, config.ASPECT_RATIO
    )

    first_frame = True
    session = GameSession(MAX_GAMES, "Floor is Lava")
    score = 0
    phase = "ACTIVE"
    last_change = time.time()
    feedback = ""

    # Initialize first game cells
    if MULTICOLOR_MODE:
        # 3 squares with colors
        all_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        active_choices = random.sample(all_cells, 3)
        available_colors = list(COLORS.keys())
        random.shuffle(available_colors)
        active_cells = {cell: COLORS[available_colors[i]] for i, cell in enumerate(active_choices)}
        safe_color_name = random.choice(available_colors)
        safe_cell = [cell for cell, col in active_cells.items() if col == COLORS[safe_color_name]][0]
    else:
        # single green square
        safe_cell = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        active_cells = safe_cell
        safe_color_name = "GREEN"

    prev_cell = safe_cell

    # --- Main loop ---
    while True:
        frame = camera_stream.read_frame()
        if frame is None:
            continue
        frame = cv2.resize(frame, (window_width, window_height))
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        trapezoid = get_trapezoid_points(w, h)
        elapsed = time.time() - last_change
        remaining = (max(3, ACTIVE_INTERVAL - score // 3) if phase == "ACTIVE" else REST_INTERVAL) - int(elapsed)

        standing_safe = check_feet_in_safe_cell(results, w, h, safe_cell, trapezoid, frame)

        if phase == "ACTIVE" and elapsed > max(3, ACTIVE_INTERVAL - score // 3):
            success = standing_safe
            feedback = "Nice!" if success else "Try Again"
            session.record_result(success)
            if success:
                score += 1
            if session.is_finished():
                break
            phase = "REST"
            last_change = time.time()
        elif phase == "REST" and elapsed > REST_INTERVAL:
            if MULTICOLOR_MODE:
                all_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
                active_choices = random.sample(all_cells, 3)
                random.shuffle(available_colors)
                active_cells = {cell: COLORS[available_colors[i]] for i, cell in enumerate(active_choices)}
                safe_color_name = random.choice(available_colors)
                safe_cell = [cell for cell, col in active_cells.items() if col == COLORS[safe_color_name]][0]
            else:
                while True:
                    new_cell = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
                    if new_cell != prev_cell:
                        safe_cell = new_cell
                        active_cells = safe_cell
                        prev_cell = safe_cell
                        break
            phase = "ACTIVE"
            feedback = ""
            last_change = time.time()

        frame = draw_grid(frame, active_cells, safe_color_name, trapezoid, MULTICOLOR_MODE)
        draw_ui(frame, phase, score, feedback, remaining, w, h, session.current_round + 1, MAX_GAMES, MULTICOLOR_MODE, safe_color_name)

        cv2.imshow(window_name, frame)

        if first_frame:
            display_manager.center_window(window_name, window_width, window_height, screen_width, screen_height)
            first_frame = False

        key = cv2.waitKey(5) & 0xFF
        if key == 32 and session.is_finished():
            break
        elif key == 27:
            cv2.destroyWindow(window_name)
            raise ExitProgram()

    # --- Summary ---
    success_count, fail_count, results_list = session.summary()
    while True:
        frame = camera_stream.read_frame()
        if frame is None:
            continue
        frame = cv2.resize(frame, (window_width, window_height))
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (w-50, h-50), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.putText(frame, f"Score: {success_count}/{MAX_GAMES}", (w//2 - 200, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 4)
        cv2.putText(frame, "Game Results:", (w//2 - 200, 180),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 0), 3)

        for idx, result in enumerate(results_list):
            color = (0, 255, 0) if result == "Success" else (0, 0, 255)
            cv2.putText(frame, f"{idx+1}: {result}", (w//2 - 200, 250 + idx*50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.putText(frame, "SPACE - next game | ESC - quit", (w//2 - 200, h-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(5) & 0xFF
        if key == 32:
            break
        elif key == 27:
            cv2.destroyWindow(window_name)
            raise ExitProgram()

    # cv2.destroyWindow(window_name)

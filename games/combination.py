import os
import cv2
import mediapipe as mp
import random
import time
import numpy as np
from utils.game import ExitProgram

# --- Settings ---
GRID_SIZE = 3
ACTIVE_INTERVAL = 15
REST_INTERVAL = 3
MAX_GAMES = 10

SAFE_PREVIEW_TIME = 3  # seconds before timer starts
MULTICOLOR_MODE = False  # True → 3 squares with colors, False → single green square
REQUIRE_BOTH_FEET = True  # default; overridden in FIREBALL_MODE
FIREBALL_MODE = True
FIREBALL_HOLD_TIME = 2  # seconds
FIREBALL_SCALE = 1.5  # multiplier for fireball size

SUMMARY_BG_PATH = "assets/ui/scoreboard/summary_background.png"
# --- Colors ---
COLORS = {
    "RED": (0, 0, 255),
    "GREEN": (0, 200, 0),
    "BLUE": (255, 0, 0)
}

# --- Mediapipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

BASE_DIR = os.path.dirname(__file__)  # directory of this file
ASSETS_DIR = os.path.join(BASE_DIR, "..","assets")

fireball_img_path = os.path.join(ASSETS_DIR, "fireball.png")
fireball_img = cv2.imread(fireball_img_path, cv2.IMREAD_UNCHANGED)

if fireball_img is None:
    raise FileNotFoundError(f"Cannot load fireball image at {fireball_img_path}")

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
            else:
                if active_cells is not None and (gx, gy) == active_cells[0]:
                    color = active_cells[1]
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.polylines(overlay, [pts], True, (0, 255, 255), 4)
                else:
                    cv2.fillPoly(overlay, [pts], (40, 40, 40))

    return cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

def spawn_fireball(w, h, trapezoid, scale=1.0):
    top_left, top_right, bottom_left, bottom_right = trapezoid
    margin = 100
    attempts = 0
    while attempts < 50:
        fx = random.randint(margin, w - margin)
        fy = random.randint(margin, int(h // 1.5) - margin) #66% top part of screen
        pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
        if cv2.pointPolygonTest(pts, (fx, fy), False) < 0:
            fireball_h, fireball_w = int(fireball_img.shape[0]*scale), int(fireball_img.shape[1]*scale)
            return {"pos": (fx, fy), "size": (fireball_w, fireball_h), "start_time": None}
        attempts += 1
    return {"pos": (margin, margin), "size": (int(fireball_img.shape[1]*scale), int(fireball_img.shape[0]*scale)), "start_time": None}

def draw_fireball(frame, fireball):
    x, y = fireball["pos"]
    fw, fh = fireball["size"]
    resized = cv2.resize(fireball_img, (fw, fh), interpolation=cv2.INTER_AREA)
    if resized.shape[2] == 4:
        alpha_s = resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            y1, y2 = max(y - fh//2,0), min(y + fh//2, frame.shape[0])
            x1, x2 = max(x - fw//2,0), min(x + fw//2, frame.shape[1])
            frame[y1:y2, x1:x2, c] = (alpha_s[0:(y2-y1),0:(x2-x1)] * resized[0:(y2-y1),0:(x2-x1),c] +
                                      alpha_l * frame[y1:y2, x1:x2, c])
    else:
        cv2.rectangle(frame, (x-fw//2, y-fh//2), (x+fw//2, y+fh//2), (0,0,255), -1)

def check_hand_on_fireball(results, fireball, w, h, frame):
    if results.pose_landmarks is None:
        fireball["start_time"] = None
        return False
    landmarks = results.pose_landmarks.landmark
    hands = [mp_pose.PoseLandmark.LEFT_INDEX, mp_pose.PoseLandmark.RIGHT_INDEX]
    x, y = fireball["pos"]
    fw, fh = fireball["size"]
    hit = False
    for hand in hands:
        px, py = int(landmarks[hand].x * w), int(landmarks[hand].y * h)
        if (x - fw//2 <= px <= x + fw//2) and (y - fh//2 <= py <= y + fh//2):
            hit = True
            cv2.circle(frame, (px, py), 15, (0, 255, 255), -1)
            break
    if hit:
        if fireball["start_time"] is None:
            fireball["start_time"] = time.time()
        elif time.time() - fireball["start_time"] >= FIREBALL_HOLD_TIME:
            return True
    else:
        fireball["start_time"] = None
    return False

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

    if phase == "PREVIEW":
        badge_img = cv2.imread(f"assets/ui/floor_is_lava/{safe_color_name.lower()}.png", cv2.IMREAD_UNCHANGED)
        if badge_img is not None:
            scale = 0.6  # adjust size
            bh, bw = badge_img.shape[:2]
            new_w = int(bw * scale)
            new_h = int(bh * scale)
            badge_img_resized = cv2.resize(badge_img, (new_w, new_h))

            # Center on screen
            x = w // 2 - new_w // 2
            y = h // 2 - new_h // 2
            overlay_rgba(frame, badge_img_resized, x, y)
        return  # skip other UI during preview

    cv2.putText(frame, f"Safe Score: {score}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Game: {game_count}/{max_games}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if phase == "ACTIVE":
        digit_img = cv2.imread(f"assets/ui/numbers/{remaining}.png", cv2.IMREAD_UNCHANGED)
        if digit_img is not None:
            dh, dw = digit_img.shape[:2]
            scale = 0.5  # adjust size
            digit_img = cv2.resize(digit_img, (int(dw*scale), int(dh*scale)))
            x, y = w//2 - digit_img.shape[1]//2, h//2 - digit_img.shape[0]//2
            overlay_rgba(frame, digit_img, x, y)
    elif phase == "REST":
        if feedback == "Nice!":
            fb_img = cv2.imread("assets/ui/feedback/nice.png", cv2.IMREAD_UNCHANGED)
        else:
            fb_img = cv2.imread("assets/ui/feedback/try_again.png", cv2.IMREAD_UNCHANGED)

        if fb_img is not None:
            # --- Bottom-left placement with safe bounds ---
            padding = 10
            scale = 0.45  # scale down image if too big
            fb_h, fb_w = fb_img.shape[:2]
            new_w = min(int(fb_w * scale), w - 2*padding)
            new_h = min(int(fb_h * scale), h - 2*padding)
            fb_img_resized = cv2.resize(fb_img, (new_w, new_h))

            x = padding
            y = h - new_h - padding

            overlay_rgba(frame, fb_img_resized, x, y)

def draw_centered_text(img, text, y, font, scale, color, thickness):
    """Draw centered text on an image at vertical position y."""
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size
    x = (img.shape[1] - text_w) // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)

def overlay_rgba(background, overlay, x, y):
    """Overlay RGBA image on BGR background at position (x, y)."""
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]

    if x+w > bw or y+h > bh:
        return  # skip if out of bounds

    alpha = overlay[:,:,3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha * overlay[:,:,c] +
            (1-alpha) * background[y:y+h, x:x+w, c]
        )

# =============================
# Main Run Function
# =============================
def run(camera_stream, display_manager, config):
    global REQUIRE_BOTH_FEET
    if FIREBALL_MODE:
        REQUIRE_BOTH_FEET = False

    window_name = config.WINDOW_NAME
    screen_width, screen_height = display_manager.get_screen_size()
    window_width, window_height = display_manager.compute_window_size(
        screen_width, screen_height, config.ASPECT_RATIO
    )

    # --- Load summary background ---
    summary_bg = cv2.imread(SUMMARY_BG_PATH, cv2.IMREAD_UNCHANGED)
    if summary_bg is not None:
        summary_bg = cv2.resize(summary_bg, (window_width, window_height))

    first_frame = True
    session = GameSession(MAX_GAMES, "Combination Game")
    score = 0
    phase = "PREVIEW"  # PREVIEW → ACTIVE → REST
    last_change = time.time()
    feedback = ""

    # Initialize game cells
    if MULTICOLOR_MODE:
        all_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        active_choices = random.sample(all_cells, 3)
        available_colors = list(COLORS.keys())
        random.shuffle(available_colors)
        active_cells = {cell: COLORS[available_colors[i]] for i, cell in enumerate(active_choices)}
        safe_color_name = random.choice(available_colors)
        safe_cell = [cell for cell, col in active_cells.items() if col == COLORS[safe_color_name]][0]
    else:
        safe_cell = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        safe_color_name = random.choice(list(COLORS.keys()))
        active_cells = (safe_cell, COLORS[safe_color_name])

    prev_cell = safe_cell

    # Initialize fireball
    fireball = spawn_fireball(window_width, window_height, get_trapezoid_points(window_width, window_height), FIREBALL_SCALE) if FIREBALL_MODE else None

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

        # --- Phase handling ---
        if phase == "PREVIEW":
            # Show safe color badge
            if elapsed > SAFE_PREVIEW_TIME:
                phase = "ACTIVE"
                last_change = time.time()

        elif phase == "ACTIVE":
            fireball_success = False
            if FIREBALL_MODE:
                draw_fireball(frame, fireball)
                fireball_success = check_hand_on_fireball(results, fireball, w, h, frame)

            if elapsed > max(3, ACTIVE_INTERVAL - score // 3):
                success = standing_safe or fireball_success
                feedback = "Nice!" if success else "Try Again"
                if success:
                    score += 1
                phase = "REST"
                last_change = time.time()

        elif phase == "REST":
            if elapsed > REST_INTERVAL:
                session.record_result(feedback == "Nice!")
                if session.is_finished():
                    success_count, fail_count, results_list = session.summary()
                    frame = summary_bg.copy() if summary_bg is not None else np.zeros((window_height, window_width, 3), dtype=np.uint8)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    title_gap = int(window_height * 0.05)  # <-- extra gap between title and Success line
                    spacing_summary = int(window_height * 0.05)  # space for Success line
                    spacing_rounds = int(window_height * 0.03)   # spacing between each Round line

                    total_lines_height = spacing_summary + spacing_rounds * len(results_list)
                    start_y = int((window_height - total_lines_height) // 2.3)

                    # --- Overlay title PNG ---
                    title_png = cv2.imread("assets/ui/scoreboard/summary_title.png", cv2.IMREAD_UNCHANGED)
                    if title_png is not None:
                        scale = 0.8  # adjust size as needed
                        th, tw = title_png.shape[:2]
                        new_w, new_h = int(tw * scale), int(th * scale)
                        title_resized = cv2.resize(title_png, (new_w, new_h))
                        x = (window_width - new_w) // 2
                        y = start_y - new_h - title_gap
                        overlay_rgba(frame, title_resized, x, y)


                    # --- Success/Total line ---
                    sf_text = f"Success: {success_count}/{MAX_GAMES}"
                    draw_centered_text(frame, sf_text, start_y, font, 1.5, (15, 125, 15), 4)

                    # --- Round results ---
                    for i, r in enumerate(results_list):
                        color = (28, 180, 13) if r == "Success" else (15, 15, 220)
                        y = start_y + spacing_summary + spacing_rounds * i
                        draw_centered_text(frame, f"Round {i+1}: {r}", y, font, 1.1, color, 4)

                    cv2.imshow(window_name, frame)

                    # Wait for space or Esc to exit
                    while True:
                        key = cv2.waitKey(10) & 0xFF
                        if key == 32 or key == 27:
                            cv2.destroyWindow(window_name)
                            raise ExitProgram()

                # --- Prepare next round ---
                prev_cell = safe_cell
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
                            safe_color_name = random.choice(list(COLORS.keys()))
                            active_cells = (safe_cell, COLORS[safe_color_name])
                            prev_cell = safe_cell
                            break

                if FIREBALL_MODE:
                    fireball = spawn_fireball(w, h, trapezoid, FIREBALL_SCALE)

                phase = "PREVIEW"
                feedback = ""
                last_change = time.time()

        # --- Draw trapezoid grid and UI ---
        frame = draw_grid(frame, active_cells, safe_color_name, trapezoid, MULTICOLOR_MODE)
        game_counter = session.current_round + 1
        draw_ui(frame, phase, score, feedback, remaining, w, h, game_counter, MAX_GAMES, MULTICOLOR_MODE, safe_color_name)

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

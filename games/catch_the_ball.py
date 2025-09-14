import os
import cv2
import mediapipe as mp
import random
import time
import numpy as np

from utils.game import ExitProgram

# --- Load Assets ---
BASE_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, "..", "assets")

fireball_img_path = os.path.join(ASSETS_DIR, "fireball.png")
fireball_img = cv2.imread(fireball_img_path, cv2.IMREAD_UNCHANGED)
if fireball_img is None:
    raise FileNotFoundError(f"Cannot load fireball image at {fireball_img_path}")

SUMMARY_BG_PATH = os.path.join(ASSETS_DIR, "ui/scoreboard/summary_background.png")

# --- Settings ---
BALL_SCALE = 1.5
ACTIVE_INTERVAL = 7
REST_INTERVAL = 3
MAX_GAMES = 10

# --- Mediapipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Shared Scoring System ---
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

# --- Utility Functions ---
def overlay_rgba(background, overlay, x, y):
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]
    if x+w > bw or y+h > bh:
        return
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha * overlay[:, :, c] + (1-alpha) * background[y:y+h, x:x+w, c]
        )

def draw_centered_text(img, text, y, font, scale, color, thickness):
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size
    x = (img.shape[1] - text_w) // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)

def draw_ball(frame, ball_pos, sprite, scale=0.2):
    bx, by = ball_pos
    sprite_h, sprite_w = sprite.shape[:2]
    new_w, new_h = int(sprite_w * scale), int(sprite_h * scale)
    sprite_resized = cv2.resize(sprite, (new_w, new_h), interpolation=cv2.INTER_AREA)

    alpha = sprite_resized[:, :, 3] / 255.0
    alpha = alpha[..., None]
    sprite_rgb = sprite_resized[:, :, :3]

    x1 = bx - new_w // 2
    y1 = by - new_h // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    h, w, _ = frame.shape
    if x1 < 0: 
        sprite_rgb = sprite_rgb[:, -x1:]
        alpha = alpha[:, -x1:]
        x1 = 0
    if y1 < 0: 
        sprite_rgb = sprite_rgb[-y1:, :]
        alpha = alpha[-y1:, :]
        y1 = 0
    if x2 > w: 
        sprite_rgb = sprite_rgb[:, :w-x1]
        alpha = alpha[:, :w-x1]
        x2 = w
    if y2 > h: 
        sprite_rgb = sprite_rgb[:h-y1, :]
        alpha = alpha[:h-y1, :]
        y2 = h

    roi = frame[y1:y2, x1:x2]
    blended = (alpha * sprite_rgb + (1-alpha) * roi).astype(np.uint8)
    frame[y1:y2, x1:x2] = blended

def check_catch(frame, results, w, h, ball_pos, radius=30):
    if not results.pose_landmarks:
        return False
    landmarks = results.pose_landmarks.landmark
    hands = [mp_pose.PoseLandmark.LEFT_INDEX, mp_pose.PoseLandmark.RIGHT_INDEX]
    bx, by = ball_pos
    hitbox_radius = int(radius * 1.5)

    for hand in hands:
        px, py = int(landmarks[hand].x * w), int(landmarks[hand].y * h)
        if (px - bx)**2 + (py - by)**2 <= hitbox_radius**2:
            cv2.circle(frame, (px, py), hitbox_radius, (0, 255, 0), 2)
            return True
        cv2.circle(frame, (px, py), hitbox_radius, (0, 0, 255), 2)
    return False

def draw_ui(frame, phase, score, feedback, remaining, game_count, max_games):
    h, w, _ = frame.shape
    cv2.putText(frame, f"Catches: {score}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"Game: {game_count}/{max_games}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if phase == "ACTIVE":
        digit_img = cv2.imread(f"assets/ui/numbers/{remaining}.png", cv2.IMREAD_UNCHANGED)
        if digit_img is not None:
            dh, dw = digit_img.shape[:2]
            scale = 0.5
            digit_img = cv2.resize(digit_img, (int(dw*scale), int(dh*scale)))
            x, y = w//2 - digit_img.shape[1]//2, h//2 - digit_img.shape[0]//2
            overlay_rgba(frame, digit_img, x, y)
    elif phase == "REST":
        if feedback == "Nice!":
            fb_img = cv2.imread("assets/ui/feedback/nice.png", cv2.IMREAD_UNCHANGED)
        else:
            fb_img = cv2.imread("assets/ui/feedback/try_again.png", cv2.IMREAD_UNCHANGED)

        if fb_img is not None:
            padding = 10
            scale = 0.45
            fb_h, fb_w = fb_img.shape[:2]
            new_w = min(int(fb_w*scale), w-2*padding)
            new_h = min(int(fb_h*scale), h-2*padding)
            fb_img_resized = cv2.resize(fb_img, (new_w, new_h))
            x = padding
            y = h - new_h - padding
            overlay_rgba(frame, fb_img_resized, x, y)

# =============================
# Main Run Function
# =============================
def run(camera_stream, display_manager, config):
    window_name = config.WINDOW_NAME
    screen_width, screen_height = display_manager.get_screen_size()
    window_width, window_height = display_manager.compute_window_size(screen_width, screen_height, config.ASPECT_RATIO)

    # --- Load summary background ---
    summary_bg = cv2.imread(SUMMARY_BG_PATH, cv2.IMREAD_UNCHANGED)
    if summary_bg is not None:
        summary_bg = cv2.resize(summary_bg, (window_width, window_height))

    session = GameSession(MAX_GAMES, "Catch the Ball")
    score = 0
    phase = "ACTIVE"
    last_change = time.time()
    feedback = ""
    h, w = window_height, window_width
    ball_pos = (random.randint(100, w-100), 0)
    first_frame = True

    while True:
        frame = camera_stream.read_frame()
        if frame is None:
            continue
        frame = cv2.resize(frame, (window_width, window_height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        elapsed = time.time() - last_change
        remaining = (ACTIVE_INTERVAL if phase=="ACTIVE" else REST_INTERVAL) - int(elapsed)

        if phase == "ACTIVE":
            by = int((elapsed / ACTIVE_INTERVAL) * (h-100))
            bx = ball_pos[0]
            draw_ball(frame, (bx, by), fireball_img, scale=BALL_SCALE)

            caught = check_catch(frame, results, w, h, (bx, by))
            if caught:
                session.record_result(True)
                feedback = "Nice!"
                score += 1
                phase = "REST"
                last_change = time.time()
            elif elapsed > ACTIVE_INTERVAL:
                session.record_result(False)
                feedback = "Try Again"
                phase = "REST"
                last_change = time.time()

        elif phase == "REST":
            if not session.is_finished() and elapsed > REST_INTERVAL:
                ball_pos = (random.randint(100, w-100), 0)
                phase = "ACTIVE"
                last_change = time.time()
                feedback = ""

        draw_ui(frame, phase, score, feedback, remaining, min(session.current_round+1, MAX_GAMES), MAX_GAMES)
        display_manager.show_frame(window_name, frame)

        if first_frame:
            display_manager.center_window(window_name, window_width, window_height, screen_width, screen_height)
            first_frame = False

        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            cv2.destroyWindow(window_name)
            raise ExitProgram()

        if session.is_finished() and phase=="REST" and elapsed>REST_INTERVAL:
            break  # exit main loop to show summary

    # --- Fancy Summary Screen (from floor_is_lava) ---
    success_count, fail_count, results_list = session.summary()
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_gap = int(window_height*0.05)
    spacing_summary = int(window_height*0.05)
    spacing_rounds = int(window_height*0.03)
    total_lines_height = spacing_summary + spacing_rounds*len(results_list)
    start_y = int((window_height - total_lines_height) // 2.3)

    while True:
        frame = camera_stream.read_frame()
        if frame is None:
            continue
        frame = cv2.resize(frame, (window_width, window_height))

        # Background
        if summary_bg is not None:
            frame = summary_bg.copy()
        else:
            frame[:] = 0

        # --- Overlay title ---
        title_png = cv2.imread(os.path.join(ASSETS_DIR, "ui/scoreboard/summary_title.png"), cv2.IMREAD_UNCHANGED)
        if title_png is not None:
            scale = 0.8
            th, tw = title_png.shape[:2]
            new_w, new_h = int(tw*scale), int(th*scale)
            title_resized = cv2.resize(title_png, (new_w, new_h))
            x = (window_width - new_w)//2
            y = start_y - new_h - title_gap
            overlay_rgba(frame, title_resized, x, y)

        # --- Success / Total ---
        draw_centered_text(frame, f"Success: {success_count}/{MAX_GAMES}", start_y, font, 1.5, (15,125,15), 4)

        # --- Round results ---
        for i, r in enumerate(results_list):
            color = (28,180,13) if r=="Success" else (15,15,220)
            y = start_y + spacing_summary + spacing_rounds*i
            draw_centered_text(frame, f"Round {i+1}: {r}", y, font, 1.1, color, 4)

        display_manager.show_frame(window_name, frame)

        key = cv2.waitKey(5) & 0xFF
        if key == 32:  # Space → exit summary
            break
        elif key == 27:
            cv2.destroyWindow(window_name)
            raise ExitProgram()

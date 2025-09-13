import os
import cv2
import mediapipe as mp
import random
import time
import numpy as np

from utils.game import ExitProgram

# --- Load Assets ---
BASE_DIR = os.path.dirname(__file__)  # directory of this file
ASSETS_DIR = os.path.join(BASE_DIR, "..","assets")

fireball_img_path = os.path.join(ASSETS_DIR, "fireball.png")
fireball_img = cv2.imread(fireball_img_path, cv2.IMREAD_UNCHANGED)

if fireball_img is None:
    raise FileNotFoundError(f"Cannot load fireball image at {fireball_img_path}")

# --- Settings ---
BALL_SCALE = 1.5
ACTIVE_INTERVAL = 2
REST_INTERVAL = 2
MAX_GAMES = 2

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
def draw_ball(frame, ball_pos, sprite, scale=0.2):
    bx, by = ball_pos
    sprite_h, sprite_w = sprite.shape[:2]
    new_w, new_h = int(sprite_w * scale), int(sprite_h * scale)
    sprite_resized = cv2.resize(sprite, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if sprite_resized.shape[2] == 4:
        alpha = sprite_resized[:, :, 3] / 255.0
        alpha = alpha[..., None]
        sprite_rgb = sprite_resized[:, :, :3]
    else:
        alpha = np.ones((new_h, new_w, 1), dtype=float)
        sprite_rgb = sprite_resized

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
    blended = (alpha * sprite_rgb + (1 - alpha) * roi).astype(np.uint8)
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
        if (px - bx) ** 2 + (py - by) ** 2 <= hitbox_radius ** 2:
            cv2.circle(frame, (px, py), hitbox_radius, (0, 255, 0), 2)
            return True
        cv2.circle(frame, (px, py), hitbox_radius, (0, 0, 255), 2)
    return False

def draw_ui(frame, phase, score, feedback, remaining, game_count, max_games):
    h, w, _ = frame.shape
    cv2.putText(frame, f"Catches: {score}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"Game: {game_count}/{max_games}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if phase == "ACTIVE":
        color = (0, 255, 255)
        if remaining <= 2 and int(time.time() * 2) % 2 == 0:
            color = (0, 0, 255)
        cv2.putText(frame, str(remaining), (w // 2 - 70, h // 2), cv2.FONT_HERSHEY_DUPLEX, 5, color, 8, cv2.LINE_AA)
    elif phase == "REST":
        cv2.putText(frame, feedback, (w // 2 - 200, h // 2), cv2.FONT_HERSHEY_DUPLEX, 3,
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

    # --- Game Session ---
    session = GameSession(MAX_GAMES, "Catch the Ball")
    score = 0
    phase = "ACTIVE"
    last_change = time.time()
    feedback = ""
    h, w = window_height, window_width
    ball_pos = (random.randint(100, w-100), 0)

    while True:
        frame = camera_stream.read_frame()
        if frame is None:
            continue
        frame = cv2.resize(frame, (window_width, window_height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        elapsed = time.time() - last_change
        remaining = (ACTIVE_INTERVAL if phase == "ACTIVE" else REST_INTERVAL) - int(elapsed)

        if phase == "ACTIVE":
            by = int((elapsed / ACTIVE_INTERVAL) * (h - 100))
            bx = ball_pos[0]
            ball_current = (bx, by)
            draw_ball(frame, ball_current, fireball_img, scale=BALL_SCALE)

            caught = check_catch(frame, results, w, h, ball_current)
            if caught:
                session.record_result(True)
                feedback = "Nice!"
                score += 1
                phase = "REST"
                last_change = time.time()
            elif elapsed > ACTIVE_INTERVAL:
                session.record_result(False)
                feedback = "Missed!"
                phase = "REST"
                last_change = time.time()

        elif phase == "REST":
            if elapsed > REST_INTERVAL:
                ball_pos = (random.randint(100, w-100), 0)
                phase = "ACTIVE"
                last_change = time.time()
                feedback = ""

        draw_ui(frame, phase, score, feedback, remaining, min(session.current_round+1, MAX_GAMES), MAX_GAMES)
        display_manager.show_frame(window_name, frame)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow(window_name)
            raise ExitProgram()

        if session.is_finished():
            break

    # --- Summary Screen ---
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
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 4)
        cv2.putText(frame, "Game Results:", (w//2 - 200, 180),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 0), 3)
        for idx, result in enumerate(results_list):
            color = (0, 255, 0) if result=="Success" else (0,0,255)
            cv2.putText(frame, f"{idx+1}: {result}", (w//2 - 200, 250 + idx*50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, "R - restart | ESC - quit", (w//2 - 200, h-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        display_manager.show_frame(window_name, frame)

        key = cv2.waitKey(5) & 0xFF
        if key == 32:
            break
        elif key == 27:
            cv2.destroyWindow(window_name)
            raise ExitProgram()

    # cv2.destroyWindow(window_name)

import cv2
import mediapipe as mp
import random
import time
import numpy as np

# --- Settings ---
GRID_SIZE = 3
ACTIVE_INTERVAL = 7
REST_INTERVAL = 3
MAX_GAMES = 5  # maximum number of safe zones to appear
INTRO_VIDEO_PATH = "intro.mp4"  # path to intro video

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


def draw_grid(frame, safe_cell, trapezoid):
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
    inside_any = False
    for foot in feet:
        px, py = int(landmarks[foot].x * w), int(landmarks[foot].y * h)
        inside = cv2.pointPolygonTest(pts, (px, py), False) >= 0
        color = (0, 255, 0) if inside else (0, 0, 255)
        cv2.circle(frame, (px, py), 10, color, -1)
        if inside:
            inside_any = True
    return inside_any


def draw_ui(frame, phase, score, feedback, remaining, w, h, game_count, max_games, show_summary=False):
    cv2.putText(frame, f"Safe Score: {score}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"Game: {game_count}/{max_games}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    if show_summary:
        return
    if phase == "ACTIVE":
        color = (0, 255, 255)
        if remaining <= 3 and int(time.time() * 2) % 2 == 0:
            color = (0, 0, 255)
        cv2.putText(frame, str(remaining), (w // 2 - 70, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 5, color, 8, cv2.LINE_AA)
    elif phase == "REST":
        cv2.putText(frame, feedback, (w // 2 - 200, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 3,
                    (0, 255, 0) if feedback == "Nice!" else (0, 0, 255), 6, cv2.LINE_AA)


def play_intro_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Intro video not found")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (720, 1280))
        cv2.putText(frame, "S - Skip Video", (500, 1200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Floor is Lava", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('s'):
            break
        if key in [27, ord('q')]:
            cap.release()
            cv2.destroyAllWindows()
            exit()
    cap.release()


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
        if key in [27, ord('q')]:
            return False
        if key == 32:
            return True


# =============================
# Main Loop
# =============================
def main():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 10)

    # --- Play intro video ---
    play_intro_video(INTRO_VIDEO_PATH)

    while True:
        if not show_start_screen(cam):
            cam.release()
            cv2.destroyAllWindows()
            return

        session = GameSession(MAX_GAMES, "Floor is Lava")
        score = 0
        safe_cell = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        prev_cell = safe_cell
        phase = "ACTIVE"
        last_change = time.time()
        feedback = ""

        while cam.isOpened():
            if session.is_finished():
                break

            ret, frame = cam.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
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
                phase = "REST"
                last_change = time.time()

            elif phase == "REST" and elapsed > REST_INTERVAL:
                while True:
                    new_cell = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
                    if new_cell != prev_cell:
                        safe_cell = new_cell
                        prev_cell = safe_cell
                        break
                phase = "ACTIVE"
                feedback = ""
                last_change = time.time()

            frame = draw_grid(frame, safe_cell, trapezoid)
            draw_ui(frame, phase, score, feedback, remaining, w, h, session.current_round + 1, MAX_GAMES)
            cv2.imshow("Floor is Lava", frame)

            key = cv2.waitKey(5) & 0xFF
            if key in [27, ord('q')]:
                cam.release()
                cv2.destroyAllWindows()
                return

        # --- Summary screen ---
        success, fail, results_list = session.summary()
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            overlay = frame.copy()
            cv2.rectangle(overlay, (50, 50), (w-50, h-50), (0,0,0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            cv2.putText(frame, f"Score: {success}/{MAX_GAMES}", (w//2 - 200, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 4)
            cv2.putText(frame, "Game Results:", (w//2 - 200, 180),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 0), 3)
            for idx, result in enumerate(results_list):
                cv2.putText(frame, f"{idx+1}: {result}", (w//2 - 200, 250 + idx*50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if result=="Success" else (0,0,255), 3)

            cv2.putText(frame, "R - restart | ESC - quit", (w//2 - 200, h-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow("Floor is Lava", frame)
            key = cv2.waitKey(5) & 0xFF
            if key == ord('r'):
                break
            if key in [27, ord('q')]:
                cam.release()
                cv2.destroyAllWindows()
                return


if __name__ == "__main__":
    main()

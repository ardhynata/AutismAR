import cv2
import numpy as np
import time
from ffpyplayer.player import MediaPlayer
from utils.game import ExitProgram

ASSETS_PATH = "assets/catch_it.mp4"


def run(camera_stream, display_manager, config):
    """
    Play a video splash screen with audio using ffpyplayer.
    Space → continue, Esc → exit.
    """
    # --- Setup window ---
    screen_width, screen_height = display_manager.get_screen_size()
    window_width, window_height = display_manager.compute_window_size(
        screen_width, screen_height, config.ASPECT_RATIO
    )
    display_manager.center_window(
        config.WINDOW_NAME,
        window_width, window_height,
        screen_width, screen_height
    )

    # --- Load video ---
    player = MediaPlayer(ASSETS_PATH, ff_opts={'sync': 'audio'})

    last_frame = None
    start_time = time.time()

    while True:
        frame, val = player.get_frame()
        if val == 'eof':
            break

        if frame is not None:
            img, pts = frame  # <-- pts = presentation timestamp in seconds

            # ✅ Convert ffpyplayer frame to numpy array
            img = np.asarray(img.to_bytearray()[0]).reshape(img.get_size()[1], img.get_size()[0], 3)

            # ffpyplayer gives RGB → convert to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            last_frame = img

            # --- Sync video with audio using pts ---
            if pts is not None:
                target_time = start_time + pts
                delay = target_time - time.time()
                if delay > 0:
                    time.sleep(delay)
        elif last_frame is not None:
            img = last_frame
        else:
            continue

        # Resize video to fit window
        img_resized = cv2.resize(img, (window_width, window_height))
        cv2.imshow(config.WINDOW_NAME, img_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space → continue
            break
        elif key == 27:  # Esc → exit program
            player = None
            cv2.destroyAllWindows()
            raise ExitProgram()

    player = None

    # Fade out last frame after video ends
    if last_frame is not None:
        fade_duration = 1.0  # seconds
        fade_steps = 30
        for i in range(fade_steps):
            alpha = 1.0 - (i + 1) / fade_steps
            fade_frame = (last_frame * alpha).astype(np.uint8)
            img_resized = cv2.resize(fade_frame, (window_width, window_height))
            cv2.imshow(config.WINDOW_NAME, img_resized)
            key = cv2.waitKey(int(fade_duration * 1000 / fade_steps)) & 0xFF
            if key == 27:  # Esc
                cv2.destroyAllWindows()
                raise ExitProgram()
            elif key == 32:  # Space
                break

        # Optionally, hold on black for a moment or until key press
        black = np.zeros_like(last_frame, dtype=np.uint8)
        img_resized = cv2.resize(black, (window_width, window_height))
        cv2.imshow(config.WINDOW_NAME, img_resized)
        key = cv2.waitKey(1000) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise ExitProgram()

    cv2.destroyAllWindows()

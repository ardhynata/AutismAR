# games/welcome_screen.py
import cv2
import os
import numpy as np
from utils.game import ExitProgram

ASSETS_PATH = os.path.join(os.path.dirname(__file__), "../assets")
BUTTON_SCALE = 0.6
BUTTON_BOTTOM_RATIO = 0.85  # vertical placement ratio (0=top, 1=bottom)


def run(camera_stream, display_manager, config, fade_duration=1.0, fps=60, button_scale=BUTTON_SCALE, button_bottom_ratio=BUTTON_BOTTOM_RATIO):
    """
    Show welcome image with a clickable start button and hover effect.
    Space → continue, click button → continue, Esc → raise ExitProgram.
    button_scale: scale factor for button size
    button_bottom_ratio: vertical placement ratio (0=top, 1=bottom)
    """
    window_name = config.WINDOW_NAME
    screen_width, screen_height = display_manager.get_screen_size()

    # Compute target window size based on ASPECT_RATIO
    window_width, window_height = display_manager.compute_window_size(
        screen_width, screen_height, config.ASPECT_RATIO
    )

    # Center the window
    display_manager.center_window(window_name, window_width, window_height, screen_width, screen_height)

    # Load welcome image
    welcome_path = os.path.join(ASSETS_PATH, "welcome_screen.png")
    if not os.path.exists(welcome_path):
        raise FileNotFoundError(f"{welcome_path} not found in assets folder")
    img = cv2.imread(welcome_path)

    # Resize and crop to cover the window
    scale = max(window_width / img.shape[1], window_height / img.shape[0])
    new_w, new_h = int(img.shape[1] * scale), int(img.shape[0] * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    x_start = (new_w - window_width) // 2
    y_start = (new_h - window_height) // 2
    base_frame = img_resized[y_start:y_start+window_height, x_start:x_start+window_width]

    # Load start button
    button_path = os.path.join(ASSETS_PATH, "start_button.png")
    if not os.path.exists(button_path):
        raise FileNotFoundError(f"{button_path} not found in assets folder")
    button_img = cv2.imread(button_path, cv2.IMREAD_UNCHANGED)  # support alpha

    # Resize button
    btn_h, btn_w = button_img.shape[:2]
    new_btn_w = int(btn_w * button_scale)
    new_btn_h = int(btn_h * button_scale)
    button_img = cv2.resize(button_img, (new_btn_w, new_btn_h), interpolation=cv2.INTER_AREA)

    # Compute button position (centered horizontally, button_bottom_ratio from top)
    btn_x = (window_width - new_btn_w) // 2
    btn_y = int(window_height * button_bottom_ratio) - new_btn_h // 2

    # Flags to detect click and hover
    clicked = [False]
    hover = [False]

    def mouse_callback(event, x, y, flags, param):
        hover[0] = btn_x <= x <= btn_x + new_btn_w and btn_y <= y <= btn_y + new_btn_h
        if event == cv2.EVENT_LBUTTONDOWN and hover[0]:
            clicked[0] = True

    cv2.setMouseCallback(window_name, mouse_callback)

    # Fade-in effect
    steps = int(fade_duration * fps)
    for alpha in np.linspace(0, 1, steps):
        frame = (base_frame * alpha).astype(np.uint8)
        overlay_button(frame, button_img, btn_x, btn_y, hover=False)
        display_manager.show_frame(window_name, frame)
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == 27:
            raise ExitProgram()

    # Show final frame until Space or click
    while True:
        frame = base_frame.copy()
        overlay_button(frame, button_img, btn_x, btn_y, hover=hover[0])
        display_manager.show_frame(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32 or clicked[0]:
            break
        elif key == 27:
            raise ExitProgram()


def overlay_button(background, button, x, y, hover=False):
    """
    Overlay PNG with alpha channel on the background at position (x, y)
    hover=True → slightly brighten button
    """
    bh, bw = background.shape[:2]
    bh2, bw2 = button.shape[:2]
    if y + bh2 > bh or x + bw2 > bw:
        return  # skip if out of bounds

    alpha_s = button[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    overlay = button[:, :, :3].astype(np.float32)
    if hover:
        overlay = np.clip(overlay * 1.3, 0, 255)  # brighten by 30%

    for c in range(3):
        background[y:y+bh2, x:x+bw2, c] = (
            alpha_s * overlay[:, :, c] + alpha_l * background[y:y+bh2, x:x+bw2, c]
        ).astype(np.uint8)

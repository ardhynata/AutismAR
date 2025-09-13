import cv2
import config
from utils.camera import CameraStream
from utils.display import DisplayManager
from utils.game import ExitProgram
from games import welcome_screen, floor_is_lava, catch_the_ball, combination, floor_is_lava_video, catch_the_ball_video, combination_video

def run():
    # Initialize camera and display services
    camera_stream = CameraStream(config.CAMERA_INDEX, config.TARGET_FPS, config.MIRROR)
    display_manager = DisplayManager()

    # Compute window size and center it once
    screen_width, screen_height = display_manager.get_screen_size()
    window_width, window_height = display_manager.compute_window_size(
        screen_width, screen_height, config.ASPECT_RATIO
    )
    display_manager.center_window(config.WINDOW_NAME, window_width, window_height, screen_width, screen_height)

    try:
        try:
            welcome_screen.run(camera_stream, display_manager, config)
            floor_is_lava_video.run(camera_stream, display_manager, config)
            floor_is_lava.run(camera_stream, display_manager, config)
            catch_the_ball_video.run(camera_stream, display_manager, config)
            catch_the_ball.run(camera_stream, display_manager, config)
            combination_video.run(camera_stream, display_manager, config)
            combination.run(camera_stream, display_manager, config)
        except ExitProgram:
            print("ESC pressed. Quitting all games...")

    finally:
        # Always release the camera
        camera_stream.release()
        cv2.destroyAllWindows()  # destroy window only once at the very end


if __name__ == "__main__":
    run()

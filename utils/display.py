import cv2
import ctypes

class DisplayManager:
    @staticmethod
    def get_screen_size():
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    @staticmethod
    def compute_window_size(screen_width, screen_height, aspect_ratio=(9,16)):
        width = int(screen_height * aspect_ratio[0] / aspect_ratio[1])
        height = screen_height
        if width > screen_width:
            width = screen_width
            height = int(width * aspect_ratio[1] / aspect_ratio[0])
        return width, height

    @staticmethod
    def center_window(window_name, window_width, window_height, screen_width, screen_height):
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        ctypes.windll.user32.MoveWindow(hwnd, x, y, window_width, window_height, True)

    @staticmethod
    def show_frame(window_name, frame):
        cv2.imshow(window_name, frame)

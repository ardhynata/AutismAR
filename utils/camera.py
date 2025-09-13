import cv2

class CameraStream:
    def __init__(self, camera_index=0, target_fps=30, mirror=False):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        self.mirror = mirror
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or target_fps

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read from camera")
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        self.cap.release()

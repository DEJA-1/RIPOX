import cv2
import os
from datetime import datetime
from src.faceDetection.detection import FaceDetector

class CameraHandler:
    _SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'model')
    _HELP_TEXT = [
        "Make photo - 's'",
        "Quit - 'q'",
        "Begin analysis - 'a'"
    ]
    WINDOW_NAME = "Preview"
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720

    def __init__(self):
        os.makedirs(self._SAVE_DIR, exist_ok=True)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
        self._analyze_mode = False
        self._face_detector = None

    def start(self):
        if not self._is_opened():
            print("Can't open the camera.")
            return

        while True:
            success, frame = self.cap.read()
            if not success:
                print("Error while fetching frame.")
                break

            if self._analyze_mode and self._face_detector:
                frame = self._face_detector.analyze(frame)

            self._show_helper_texts(frame)
            cv2.imshow(self.WINDOW_NAME, frame)
            self._handle_key(frame)

    def _handle_key(self, frame):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self._stop_camera()
        elif key == ord('s'):
            self._save_frame(frame)
        elif key == ord('a'):
            self._begin_analysis()

    def _stop_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def _show_helper_texts(self, frame):
        for i, line in enumerate(self._HELP_TEXT):
            cv2.putText(frame, line, (10, 25 + i * 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def _is_opened(self):
        return self.cap.isOpened()

    def _save_frame(self, _):
        ret, clean_frame = self.cap.read()
        if not ret:
            print("Error while saving frame.")
            return
        filename = os.path.join(
            self._SAVE_DIR, f"wzorzec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, clean_frame)
        print(f"Saved: {filename}")

    def _begin_analysis(self):
        if not self._face_detector:
            self._face_detector = FaceDetector()
        self._analyze_mode = not self._analyze_mode
        print("Analysis mode:", "ON" if self._analyze_mode else "OFF")


if __name__ == "__main__":
    camera = CameraHandler()
    camera.start()

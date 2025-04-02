import cv2
import os
from datetime import datetime
from urllib.parse import urlparse
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

    def __init__(self, source=0):
        os.makedirs(self._SAVE_DIR, exist_ok=True)
        self._analyze_mode = False
        self._face_detector = None
        self.source = source
        self.cap = self._init_video_source(source)
        self._running = True

    def _init_video_source(self, source):
        if isinstance(source, int):
            cap = cv2.VideoCapture(source)
        elif isinstance(source, str):
            if source.startswith(('http://', 'rtsp://', 'https://')):
                cap = cv2.VideoCapture(source)
            elif os.path.isfile(source):
                cap = cv2.VideoCapture(source)
            else:
                raise ValueError(f"Nieprawidłowe źródło: {source}")
        else:
            raise TypeError("Źródło musi być int (indeks kamery) lub str (ścieżka/URL)")

        if not cap.isOpened():
            raise RuntimeError(f"Nie można otworzyć źródła: {source}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
        return cap

    def start(self):
        if not self._is_opened():
            print("Nie można otworzyć źródła wideo. Sprawdź ścieżkę/URL.")
            return

        while self._running:  # Teraz sprawdzamy flagę _running
            success, frame = self.cap.read()

            if not success:
                if isinstance(self.source, str) and not self.source.startswith(('http://', 'rtsp://', 'https://')):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("Błąd odczytu klatki. Kończenie...")
                    break

            if self._analyze_mode and self._face_detector:
                frame = self._face_detector.analyze(frame)

            self._show_helper_texts(frame)
            cv2.imshow(self.WINDOW_NAME, frame)

            self._handle_key()

        self._cleanup()

    def _handle_key(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self._running = False
        elif key == ord('s'):
            self._save_frame()
        elif key == ord('a'):
            self._begin_analysis()

    def _cleanup(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

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

        filename = os.path.abspath(os.path.join(
            self._SAVE_DIR,
            f"wzorzec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        ))


        try:
            is_saved = cv2.imwrite(filename, clean_frame)

            if not is_saved:
                raise ValueError(f"Nie udało się zapisać pliku {filename}. Sprawdź ścieżkę i uprawnienia.")

            print(f"Pomyślnie zapisano obraz do {filename}")

        except cv2.error as cv_err:
            print(f"Błąd OpenCV: {cv_err}")
        except IOError as io_err:
            print(f"Błąd we/wy: {io_err} - problem z zapisem na dysku")
        except Exception as e:
            print(f"Inny nieoczekiwany błąd: {e}")


        print(f"Saved: {filename}")

    def _begin_analysis(self):
        if not self._face_detector:
            self._face_detector = FaceDetector()
            recognizer = self._face_detector._get_recognizer()
            known_faces_dir = os.path.join(os.path.dirname(__file__), '..', 'known_faces')
            if os.path.exists(known_faces_dir):
                import glob
                for img_path in glob.glob(os.path.join(known_faces_dir, "*.jpg")):
                    name = os.path.splitext(os.path.basename(img_path))[0]
                    face_img = cv2.imread(img_path)
                    if face_img is not None:
                        recognizer.register_face(face_img, name)
        self._analyze_mode = not self._analyze_mode


if __name__ == "__main__":
    camera = CameraHandler()
    camera.start()
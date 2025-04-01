import os
import sys
import cv2
from src.faceDetection.face_sdk.face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from src.faceDetection.face_sdk.face_sdk.core.model_handler.face_detection.FaceDetModelHandler import \
    FaceDetModelHandler

# Usunięto bezpośredni import FaceRecognition tutaj

sdk_root = os.path.join(os.path.dirname(__file__), "face_sdk", "face_sdk")
if sdk_root not in sys.path:
    sys.path.insert(0, sdk_root)


class FaceDetector:
    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(__file__),
            "face_sdk/face_sdk",
            "models"
        )
        model_category = "face_detection"
        model_name = "face_detection_1.0"

        loader = FaceDetModelLoader(model_path, model_category, model_name)
        model, cfg = loader.load_model()
        self.handler = FaceDetModelHandler(model, "cpu", cfg)
        self.recognizer = None  # Inicjalizacja opóźniona

    def _get_recognizer(self):
        """Leniwe ładowanie recognizera tylko gdy potrzebny"""
        if self.recognizer is None:
            from src.faceRecognition.face_recognition import FaceRecognition
            self.recognizer = FaceRecognition()
        return self.recognizer

    def detect_faces(self, frame):
        """Wykrywanie twarzy bez zależności od recognizera"""
        bboxes = self.handler.inference_on_image(frame)
        return bboxes

    def draw_faces(self, frame, bboxes):
        """Rysowanie twarzy z opcjonalnym rozpoznawaniem"""
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            face_crop = frame[y1:y2, x1:x2]

            # Rozpoznawanie tylko jeśli recognizer jest potrzebny
            label = "Twarz"
            if hasattr(self, 'recognizer') and self.recognizer is not None:
                try:
                    identity = self._get_recognizer().recognize_face(face_crop)
                    label = identity if identity else "Nieznana"
                except Exception as e:
                    print(f"Błąd rozpoznawania: {e}")
                    label = "Błąd rozpoznania"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame

    def analyze(self, frame):
        """Główna metoda analizy"""
        bboxes = self.detect_faces(frame)
        return self.draw_faces(frame, bboxes)


if __name__ == "__main__":
    # Testowanie niezależne
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.analyze(frame)
        cv2.imshow("Face Detection", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
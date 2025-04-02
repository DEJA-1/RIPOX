import glob
import os
import sys
import cv2
import numpy as np
import pickle
from src.faceDetection.face_sdk.face_sdk.core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from src.faceDetection.face_sdk.face_sdk.core.model_handler.face_recognition.FaceRecModelHandler import \
    FaceRecModelHandler

sdk_root = os.path.join(os.path.dirname(__file__), "face_sdk", "face_sdk")
if sdk_root not in sys.path:
    sys.path.insert(0, sdk_root)


class FaceRecognition:
    def __init__(self, db_path='face_database.pkl'):
        model_path = os.path.join(
            os.path.dirname(__file__),
            "face_sdk/face_sdk",
            "models"
        )
        loader = FaceRecModelLoader(model_path, "face_recognition", "face_recognition_1.0")
        model, cfg = loader.load_model()
        self.handler = FaceRecModelHandler(model, "cpu", cfg)

        self.db_path = db_path
        self.known_faces = self._load_database()

    def _load_database(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_database(self):
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.known_faces, f)

    def register_face(self, face_image, name):
        embedding = self._get_face_embedding(face_image)
        if embedding is not None:
            if name not in self.known_faces:
                self.known_faces[name] = []
            self.known_faces[name].append(embedding)
            self._save_database()
            return True
        return False

    def _get_face_embedding(self, face_image):
        try:
            if len(face_image.shape) == 2:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
            return self.handler.inference_on_image(face_image)
        except Exception as e:
            print(f"Błąd przetwarzania twarzy: {e}")
            return None

    def recognize_face(self, frame, bboxes, threshold=0.6):
        results = []
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            face = frame[y1:y2, x1:x2]

            unknown_embedding = self._get_face_embedding(face)
            if unknown_embedding is None:
                continue

            identity = None
            max_similarity = -1

            for name, embeddings in self.known_faces.items():
                for known_embedding in embeddings:
                    similarity = np.dot(unknown_embedding, known_embedding.T)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        identity = name if similarity > threshold else None

            results.append({
                'box': box,
                'embedding': unknown_embedding,
                'identity': identity,
                'similarity': max_similarity
            })

        return results

    def draw_recognition_results(self, frame, results):
        for result in results:
            x1, y1, x2, y2 = map(int, result['box'][:4])
            color = (0, 255, 0) if result['identity'] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = result['identity'] if result['identity'] else f"Nieznany ({result['similarity']:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame


if __name__ == "__main__":
    recognizer = FaceRecognition()
    known_faces_dir = os.path.join(os.path.dirname(__file__), '..', 'known_faces')  # Ścieżka: src/known_faces
    if os.path.exists(known_faces_dir):
        for img_path in glob.glob(os.path.join(known_faces_dir, "*.jpg")):
            name = os.path.splitext(os.path.basename(img_path))[0]
            face_img = cv2.imread(img_path)
            if face_img is not None:
                recognizer.register_face(face_img, name)
import json
import os
import sys
import cv2
import numpy as np
import glob
import torch
import pygame
from sympy import false
from collections import defaultdict

from src.configurator.configurator import Configurator
from src.faceDetection.face_sdk.face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from src.faceDetection.face_sdk.face_sdk.core.model_handler.face_detection.FaceDetModelHandler import \
    FaceDetModelHandler
from src.faceDetection.face_sdk.face_sdk.core.model_loader.face_alignment.FaceAlignModelLoader import \
    FaceAlignModelLoader
from src.faceDetection.face_sdk.face_sdk.core.model_handler.face_alignment.FaceAlignModelHandler import \
    FaceAlignModelHandler
from src.faceDetection.face_sdk.face_sdk.core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from src.faceDetection.face_sdk.face_sdk.core.model_handler.face_recognition.FaceRecModelHandler import \
    FaceRecModelHandler
from src.faceDetection.face_sdk.face_sdk.core.image_cropper.arcface_cropper.FaceRecImageCropper import \
    FaceRecImageCropper

sdk_root = os.path.join(os.path.dirname(__file__), "face_sdk", "face_sdk")
if sdk_root not in sys.path:
    sys.path.insert(0, sdk_root)


class FaceRecognition:
    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(__file__),
            "face_sdk/face_sdk",
            "models"
        )

        det_loader = FaceDetModelLoader(model_path, "face_detection", "face_detection_1.0")
        det_model, det_cfg = det_loader.load_model()
        self.det_handler = FaceDetModelHandler(det_model, "cpu", det_cfg)

        align_loader = FaceAlignModelLoader(model_path, "face_alignment", "face_alignment_1.0")
        align_model, align_cfg = align_loader.load_model()
        align_model = align_model.to('cpu')
        self.align_handler = FaceAlignModelHandler(align_model, "cpu", align_cfg)

        rec_loader = FaceRecModelLoader(model_path, "face_recognition", "face_recognition_1.0")
        rec_model, rec_cfg = rec_loader.load_model()
        rec_model = rec_model.to('cpu')
        self.rec_handler = FaceRecModelHandler(rec_model, "cpu", rec_cfg)

        self.face_cropper = FaceRecImageCropper()
        self.known_faces = defaultdict(list)
        self.avg_embeddings = {}

    def _preprocess_face(self, frame, box):
        try:
            landmarks = self.align_handler.inference_on_image(frame, box)
            landmarks_list = []
            for (x, y) in landmarks.astype(np.int32):
                landmarks_list.extend((x, y))

            cropped_image = self.face_cropper.crop_image_by_mat(frame, landmarks_list)

            if cropped_image is None or cropped_image.size == 0:
                return None

            if len(cropped_image.shape) == 2:
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)

            cropped_image = cropped_image.astype('float32') / 255.0

            return cropped_image
        except Exception as e:
            print(f"Face processing error: {e}")
            return None

    def register_face(self, frame, name, box):
        try:
            cropped_face = self._preprocess_face(frame, box)
            if cropped_face is None:
                return False

            embedding = self.rec_handler.inference_on_image(cropped_face)

            if torch.is_tensor(embedding):
                embedding = embedding.cpu().numpy()
            elif not isinstance(embedding, np.ndarray):
                raise ValueError("Unknown embedding type")

            self.known_faces[name].append(embedding)

            self._update_avg_embedding(name)

            print(f"Registered face: {name} (total samples: {len(self.known_faces[name])})")
            return True
        except Exception as e:
            print(f"Registering face error: {e}")
            return False

    def _update_avg_embedding(self, name):
        if name in self.known_faces and len(self.known_faces[name]) > 0:
            self.avg_embeddings[name] = np.mean(self.known_faces[name], axis=0)

    def _scale_similarity(self, similarity):
        min_val = 0.9965
        max_val = 0.999

        if similarity <= min_val:
            return 0.0
        elif similarity >= max_val:
            return similarity
        else:
            return (similarity - min_val) / (max_val - min_val) * max_val

    def recognize_face(self, frame, box, threshold):
        try:
            cropped_face = self._preprocess_face(frame, box)
            if cropped_face is None:
                return {"identity": "Unknown", "similarity": 0}

            unknown_embedding = self.rec_handler.inference_on_image(cropped_face)
            unknown_embedding = _normalize(unknown_embedding)

            best_match = "Unknown"
            max_similarity = 0

            for name, avg_embedding in self.avg_embeddings.items():
                if avg_embedding is not None:
                    avg_embedding = _normalize(avg_embedding)
                    similarity = np.dot(unknown_embedding, avg_embedding.T)

                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = name

            scaled_similarity = self._scale_similarity(max_similarity)

            if scaled_similarity >= threshold:
                identity = best_match
            else:
                identity = "Unknown"

            print(f"Best match: {identity}, Similarity: {scaled_similarity:.4f}, Threshold: {threshold}")

            return {"identity": identity, "similarity": float(scaled_similarity)}

        except Exception as e:
            print(f"CRITICAL ERROR in recognition: {e}")
            return {"identity": "Unknown", "similarity": 0}


def _normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


class FaceDetector:
    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(__file__),
            "face_sdk/face_sdk",
            "models"
        )

        det_loader = FaceDetModelLoader(model_path, "face_detection", "face_detection_1.0")
        det_model, det_cfg = det_loader.load_model()
        self.det_handler = FaceDetModelHandler(det_model, "cpu", det_cfg)

        self.recognizer = FaceRecognition()

        self._load_known_faces()
        self.user_config = Configurator.load_user_config()
        self._last_recognized = False

    def _load_known_faces(self):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'model'))
        if os.path.exists(data_dir):
            print(f"Loading known faces from: {data_dir}")

            for person_dir in glob.glob(os.path.join(data_dir, "*")):
                if os.path.isdir(person_dir):
                    name = os.path.basename(person_dir)
                    print(f"Processing person: {name}")

                    for img_path in glob.glob(os.path.join(person_dir, "*.jpg")):
                        print(f"  Loading image: {os.path.basename(img_path)}")

                        img = cv2.imread(img_path)
                        if img is not None:
                            bboxes = self.detect_faces(img)
                            if bboxes.size > 0:
                                if self.recognizer.register_face(img, name, bboxes[0]):
                                    print(f"  Registered: {os.path.basename(img_path)}")
                                else:
                                    print(f"  Could not register: {os.path.basename(img_path)}")
                            else:
                                print(f"  Could not detect face in: {os.path.basename(img_path)}")
                        else:
                            print(f"  Cannot load: {os.path.basename(img_path)}")
        else:
            print(f"No known_faces folder: {data_dir}")

    def detect_faces(self, frame):
        return self.det_handler.inference_on_image(frame)

    def draw_glasses(self, frame, bboxes):
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            face_width = x2 - x1
            face_height = y2 - y1

            eye_height = y1 + face_height // 3
            rect_w = face_width // 3
            rect_h = face_height // 6
            spacing = face_width // 10

            left_x = x1 + spacing
            right_x = left_x + rect_w
            top_y = eye_height
            bottom_y = top_y + rect_h
            cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (0, 0, 0), -1)

            left_x = x2 - spacing - rect_w
            right_x = left_x + rect_w
            cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (0, 0, 0), -1)

            middle_x1 = x1 + face_width // 2 - rect_w // 10
            middle_x2 = x1 + face_width // 2 + rect_w // 10
            cv2.rectangle(frame, (middle_x1, top_y + rect_h // 3), (middle_x2, top_y + 2 * rect_h // 3), (0, 0, 0), -1)

        return frame
    def analyze(self, frame):
        crop_cfg = self.user_config.get("frame_crop", {})
        x = crop_cfg.get("x", 0)
        y = crop_cfg.get("y", 0)
        w = crop_cfg.get("width", frame.shape[1])
        h = crop_cfg.get("height", frame.shape[0])
        frame = frame[y:y + h, x:x + w]

        bboxes = self.detect_faces(frame)
        if bboxes.size == 0:
            return frame

        threshold = float(self.user_config.get("detection", {}).get("confidence_threshold", 0.9991))

        results = [self.recognizer.recognize_face(frame, box, threshold) for box in bboxes]

        frame = self.draw_faces(frame, bboxes, results)

        if self.user_config.get("overlay", {}).get("show_lines", False):
            h, w = frame.shape[:2]
            cv2.line(frame, (0, 0), (w, h), (255, 0, 0), 1)
            cv2.line(frame, (w, 0), (0, h), (255, 0, 0), 1)

        return frame

    def draw_faces(self, frame, bboxes, recognition_results):
        any_recognized = false
        for box, result in zip(bboxes, recognition_results):
            x1, y1, x2, y2 = map(int, box[:4])
            recognized = result['identity'] != "Unknown"
            color = (0, 255, 0) if result['identity'] != "Unknown" else (0, 0, 255)
            label = f"{result['identity']} ({result['similarity']:.4f})"

            if self.user_config.get("alert", {}).get("show_rectangle", True):
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if self.user_config.get("alert", {}).get("show_alert", False):
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if recognized:
                any_recognized = True

            if any_recognized and not self._last_recognized:
                if self.user_config.get("alert", {}).get("play_sound", False):
                    self._play_sound()

            self._last_recognized = any_recognized
        return frame
    def _play_sound(self):
        sound_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'utils', 'detect.mp3'
        ))
        pygame.mixer.init()
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()

if __name__ == "__main__":
    print("System Initialization...")
    detector = FaceDetector()

    print("Starting the camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera!")
        exit()

    print("Start processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error while fetching frame")
            break

        try:
            result = detector.analyze(frame)
            cv2.imshow("Face detection", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Processing error: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("System closed")

import os

import cv2
import mediapipe as mp

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, 'models', 'face_landmarker.task')
MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/'
             'face_landmarker/float16/latest/face_landmarker.task')

LEGACY = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh')

POSE_POINTS = [1, 10, 33, 263, 61, 291, 152]
EYE_POINTS = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]
IRIS_POINTS = [468, 473]

FACEMESH_CONTOURS = [
    [270, 409], [176, 149], [37, 0], [84, 17], [318, 324], [293, 334],
    [386, 385], [7, 163], [33, 246], [17, 314], [374, 380], [251, 389],
    [390, 373], [267, 269], [295, 285], [389, 356], [173, 133], [33, 7],
    [377, 152], [158, 157], [405, 321], [54, 103], [263, 466], [324, 308],
    [67, 109], [409, 291], [157, 173], [454, 323], [388, 387], [78, 191],
    [148, 176], [311, 310], [39, 37], [249, 390], [144, 145], [402, 318],
    [80, 81], [310, 415], [153, 154], [384, 398], [397, 365], [234, 127],
    [103, 67], [282, 295], [338, 297], [378, 400], [127, 162], [321, 375],
    [375, 291], [317, 402], [81, 82], [154, 155], [91, 181], [334, 296],
    [297, 332], [269, 270], [150, 136], [109, 10], [356, 454], [58, 132],
    [312, 311], [152, 148], [415, 308], [161, 160], [296, 336], [65, 55],
    [61, 146], [78, 95], [380, 381], [398, 362], [361, 288], [246, 161],
    [162, 21], [0, 267], [82, 13], [132, 93], [314, 405], [10, 338],
    [178, 87], [387, 386], [381, 382], [70, 63], [61, 185], [14, 317],
    [105, 66], [300, 293], [382, 362], [88, 178], [185, 40], [46, 53],
    [284, 251], [400, 377], [136, 172], [323, 361], [13, 312], [21, 54],
    [172, 58], [373, 374], [163, 144], [276, 283], [53, 52], [365, 379],
    [379, 378], [146, 91], [263, 249], [283, 282], [87, 14], [145, 153],
    [155, 133], [93, 234], [66, 107], [95, 88], [159, 158], [52, 65],
    [332, 284], [40, 39], [191, 80], [63, 105], [181, 84], [466, 388],
    [149, 150], [288, 397], [160, 159], [385, 384],
]


def _download_model():
    import urllib.request
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print('Downloading face_landmarker.task model...')
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print('Model downloaded.')


def create_face_mesh(num_faces=2, refine=True):
    if LEGACY:
        return _LegacyTracker(num_faces, refine)
    if not os.path.isfile(MODEL_PATH):
        _download_model()
    return _TasksTracker(num_faces)


def _draw_mesh(frame, landmarks, start, end):
    image_h, image_w = frame.shape[:2]
    p1 = (int(landmarks[start].x * image_w), int(landmarks[start].y * image_h))
    p2 = (int(landmarks[end].x * image_w), int(landmarks[end].y * image_h))
    cv2.line(frame, p1, p2, (0, 150, 255), 1)


class _LegacyTracker:
    def __init__(self, num_faces, refine):
        self._fm = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=num_faces,
            refine_landmarks=refine,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def process(self, frame_rgb):
        res = self._fm.process(frame_rgb)
        if res is None or not res.multi_face_landmarks:
            return []
        return [face for face in res.multi_face_landmarks]

    def draw(self, frame, faces):
        drawing = mp.solutions.drawing_utils
        spec_landmark = drawing.DrawingSpec(color=(0, 200, 0), thickness=1,
                                            circle_radius=1)
        spec_connection = drawing.DrawingSpec(color=(0, 150, 255), thickness=1)
        for face in faces:
            drawing.draw_landmarks(
                frame, face, FACEMESH_CONTOURS,
                landmark_drawing_spec=spec_landmark,
                connection_drawing_spec=spec_connection)


class _TasksTracker:
    def __init__(self, num_faces):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (FaceLandmarker,
                                                   FaceLandmarkerOptions,
                                                   RunningMode)
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_faces=num_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False)
        self._lm = FaceLandmarker.create_from_options(options)
        self._ts = 0

    def process(self, frame_rgb):
        self._ts += 33000
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = self._lm.detect_for_video(img, self._ts)
        if res is None or not res.face_landmarks:
            return []
        return [face for face in res.face_landmarks]

    def draw(self, frame, faces):
        for face in faces:
            landmarks = face.landmark
            seen = set()
            for a, b in FACEMESH_CONTOURS:
                _draw_mesh(frame, landmarks, a, b)
                seen.add(a)
                seen.add(b)
            for idx in sorted(seen):
                p = landmarks[idx]
                cv2.circle(frame, (int(p.x * frame.shape[1]),
                                   int(p.y * frame.shape[0])),
                           1, (0, 200, 0), -1)
import cv2
import numpy as np
from collections import deque

import mp_face
from mp_face import create_face_mesh

face_mesh = create_face_mesh(num_faces=1, refine=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

score_history = deque(maxlen=10)
distraction = 0

def eye_aspect_ratio(landmarks, eye_points, image_w, image_h):
    p = []
    for idx in eye_points:
        lm = landmarks[idx]
        x, y = int(lm.x * image_w), int(lm.y * image_h)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1) 
        p.append((x, y))

    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    ear = (A + B) / (2.0 * C)
    return ear

def is_blinking(ear, threshold=0.2):
    return ear < threshold

def get_head_pose_score(landmarks, image_w, image_h):
    nose = landmarks[1]
    x = nose.x * image_w
    y = nose.y * image_h
    d = np.linalg.norm(np.array([x - image_w / 2, y - image_h / 2]))
    if d < 0.3 * image_w:  
        return 1.0
    return 0.0

def get_gaze_score(landmarks, image_w, image_h):
    left_iris = landmarks[468]
    right_iris = landmarks[473]
    avg_x = (left_iris.x + right_iris.x) / 2.0
    if 0.5 < avg_x < 0.7:
        return 1.0  
    return 0.0     

def compute_concentration_score(gaze, head_pose, blink):
    score = 0.4 * gaze + 0.4 * head_pose + 0.2 * (0 if blink else 1)
    return round(score * 100, 2)

def bar(score, frame):
    """Enhanced visual bar for concentration level"""
    bar_width = 200
    bar_height = 30
    bar_x = 30
    bar_y = 100
    
    cv2.rectangle(frame, (bar_x, bar_y), 
                 (bar_x + bar_width, bar_y + bar_height), 
                 (50, 50, 50), -1)
    
    fill_width = int(score * bar_width / 100)
    color = (0, 255, 0) if score > 40 else (0, 100, 255)
    cv2.rectangle(frame, (bar_x, bar_y), 
                 (bar_x + fill_width, bar_y + bar_height), 
                 color, -1)
 
    cv2.rectangle(frame, (bar_x, bar_y), 
                 (bar_x + bar_width, bar_y + bar_height), 
                 (200, 200, 200), 2)
    
    cv2.putText(frame, f"{score}%", 
               (bar_x + bar_width + 10, bar_y + bar_height//2 + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Please check camera permissions or index.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame from webcam.")
        break

    ui_bg = frame.copy()
    cv2.rectangle(ui_bg, (0, 0), (frame.shape[1], 150), (30, 30, 30), -1)
    cv2.addWeighted(ui_bg, 0.6, frame, 0.4, 0, frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_h, image_w, _ = frame.shape
    faces = face_mesh.process(frame_rgb)

    if faces:
        face_mesh.draw(frame, faces)
        for face in faces:
            landmarks = face.landmark
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, image_w, image_h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, image_w, image_h)
            avg_ear = (left_ear + right_ear) / 2

            blink = is_blinking(avg_ear)

            gaze_score = get_gaze_score(landmarks, image_w, image_h)
            head_score = get_head_pose_score(landmarks, image_w, image_h)
            concentration = compute_concentration_score(gaze_score, head_score, blink)

            score_history.append(concentration)
            smooth_score = int(np.mean(score_history))
            bar(smooth_score, frame)

            cv2.putText(frame, f"Concentration: {smooth_score}%", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if blink:
                cv2.putText(frame, "BLINKING", (30, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)

            if smooth_score < 40:
                distraction += 1
                cv2.putText(frame, f"Distraction: {distraction}", (30, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
                if distraction > 1000:
                    distraction = 0
                    print('turn off')
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f"FPS: {fps:.1f}", (image_w - 120, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
    
    status_color = (0, 255, 0) if distraction == 0 else (0, 100, 255)
    cv2.circle(frame, (image_w - 30, 70), 15, status_color, -1)
    cv2.putText(frame, "ACTIVE" if distraction == 0 else "DISTRACTED", 
               (image_w - 120, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Concentration Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
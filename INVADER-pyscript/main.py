import cv2
import mediapipe as mp
import numpy as np
import time

# --- Constants and Settings ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

CAMERA_ID = 0
YAW_THRESHOLD = 20
PITCH_THRESHOLD = 15
UNFOCUSED_TIME_LIMIT = 3 # seconds

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_FOCUSED = (0, 255, 0)
COLOR_UNFOCUSED = (0, 0, 255)
COLOR_PAUSED = (0, 255, 255) # Yellow
COLOR_BREAK = (255, 165, 0) # A nice blue/orange for the break state

# --- Pomodoro Settings ---
WORK_MINUTES = 1
BREAK_MINUTES = 5

# --- State Variables ---
session_state = "WORK" # Can be "WORK" or "BREAK"
remaining_seconds = WORK_MINUTES * 60
is_paused = False
unfocused_start_time = None
last_time = time.time()

# --- Main Program ---
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"FATAL ERROR: Cannot open camera with ID {CAMERA_ID}.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    img_h, img_w, img_c = frame.shape
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_time = time.time()
    delta_time = current_time - last_time
    last_time = current_time

    # --- Keyboard Input ---
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' '): # Spacebar to pause/resume
        is_paused = not is_paused

    # --- Timer Countdown and Session Switching ---
    if not is_paused:
        remaining_seconds -= delta_time

    if remaining_seconds <= 0:
        if session_state == "WORK":
            session_state = "BREAK"
            remaining_seconds = BREAK_MINUTES * 60
        else: # Was "BREAK"
            session_state = "WORK"
            remaining_seconds = WORK_MINUTES * 60
        # When a session ends, we should un-pause the timer
        is_paused = False

    # --- Focus Detection ---
    status = "UNFOCUSED"
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        # (Head pose estimation logic...)
        model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])
        image_points = np.array([(face_landmarks.landmark[1].x * img_w, face_landmarks.landmark[1].y * img_h), (face_landmarks.landmark[152].x * img_w, face_landmarks.landmark[152].y * img_h), (face_landmarks.landmark[263].x * img_w, face_landmarks.landmark[263].y * img_h), (face_landmarks.landmark[33].x * img_w, face_landmarks.landmark[33].y * img_h), (face_landmarks.landmark[287].x * img_w, face_landmarks.landmark[287].y * img_h), (face_landmarks.landmark[57].x * img_w, face_landmarks.landmark[57].y * img_h)], dtype="double")
        cam_matrix = np.array([[float(img_w), 0, img_h / 2], [0, float(img_w), img_w / 2], [0, 0, 1]])
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs)
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(np.hstack((rot_mat, trans_vec)))
        yaw = angles[1]
        pitch = angles[0]
        if abs(yaw) < YAW_THRESHOLD and abs(pitch) < PITCH_THRESHOLD:
            status = "FOCUSED"
    
    # --- Auto-Pause Logic (Only during WORK session) ---
    if session_state == "WORK":
        if status == "FOCUSED":
            # Don't auto-resume if manually paused
            if unfocused_start_time is not None:
                is_paused = False
            unfocused_start_time = None 
        else: # UNFOCUSED
            if not is_paused: # Don't restart the timer if already manually paused
                if unfocused_start_time is None:
                    unfocused_start_time = time.time()
                elif time.time() - unfocused_start_time > UNFOCUSED_TIME_LIMIT:
                    is_paused = True
    else: # During BREAK, timer never auto-pauses
        unfocused_start_time = None


    # --- UI Drawing ---
    # Session State and Focus Status
    session_color = COLOR_BREAK if session_state == "BREAK" else (COLOR_FOCUSED if status == "FOCUSED" else COLOR_UNFOCUSED)
    cv2.putText(frame, session_state, (30, 50), FONT, 1.5, session_color, 3)
    
    if results.multi_face_landmarks:
        mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)

    # Timer
    mins = int(remaining_seconds // 60)
    secs = int(remaining_seconds % 60)
    timer_text = f"{mins:02d}:{secs:02d}"
    cv2.putText(frame, timer_text, (30, 120), FONT, 2, (255, 255, 255), 3)

    # Paused indicator
    if is_paused:
        cv2.putText(frame, "PAUSED", (30, 180), FONT, 1.5, COLOR_PAUSED, 3)

    cv2.imshow('INVADER', frame)

cap.release()
cv2.destroyAllWindows()
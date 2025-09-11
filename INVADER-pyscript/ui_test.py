import cv2
import time

# --- Timer Settings ---
WORK_MINUTES = 25
BREAK_MINUTES = 5

# --- UI Settings ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
STATE_POSITION = (50, 50)
TIMER_POSITION = (50, 120)
FONT_SCALE_STATE = 1
FONT_SCALE_TIMER = 2
COLOR_ACTIVE = (0, 255, 0) # Green
THICKNESS = 3

# Initialize the webcam.
CAMERA_ID = 0
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"FATAL ERROR: Cannot open camera with ID {CAMERA_ID}.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)

    # --- Draw the UI Elements ---

    # 1. Define the current state and timer text (static for now)
    session_state = "WORK"
    timer_text = f"{WORK_MINUTES:02d}:00"

    # 2. Draw the session state text
    cv2.putText(frame, 
                session_state, 
                STATE_POSITION, 
                FONT, 
                FONT_SCALE_STATE, 
                COLOR_ACTIVE, 
                THICKNESS)

    # 3. Draw the timer text
    cv2.putText(frame, 
                timer_text, 
                TIMER_POSITION, 
                FONT, 
                FONT_SCALE_TIMER, 
                COLOR_ACTIVE, 
                THICKNESS)
    
    # --- End of UI Drawing ---

    cv2.imshow('INVADER', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
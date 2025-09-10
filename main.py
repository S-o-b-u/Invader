import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing utilities.
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialize the webcam. The '0' indicates the default webcam.
cap = cv2.VideoCapture(0)

# Start an infinite loop to continuously capture frames from the webcam.
while cap.isOpened():

    # Read a frame from the webcam.
    # 'success' will be True if the frame was read correctly.
    # 'frame' is the actual image data.
    success, frame = cap.read()
    # If a frame was successfully captured...
    if success:
        # Display the frame in a window named "Invader".
        cv2.imshow("Invader", frame)

    # Flip the frame horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    frame.flags.writeable = False
    results = face_mesh.process(frame)
    frame.flags.writeable = True

    # Convert the image color back from RGB to BGR.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # If a face is detected, draw the face mesh landmarks.
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # Display the frame in a window named "INVADER".
    cv2.imshow('INVADER', frame)

    # If the key pressed is 'q', break out of the loop.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# After the loop ends, release the webcam.
cap.release()
# Close all the windows created by OpenCV.
cv2.destroyAllWindows()
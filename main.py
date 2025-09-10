import cv2
import mediapipe as mp
import numpy as np


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
    if not success:
        print("Ignoring empty camera frame.")
        continue

     # Get the frame dimensions.
    img_h, img_w, img_c = frame.shape
    
    # Flip the frame horizontally for a selfie-view display.
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find face landmarks.
    results = face_mesh.process(rgb_frame)

    # If a face is detected...
    if results.multi_face_landmarks:
        # We'll use the first detected face.
        face_landmarks = results.multi_face_landmarks[0]
        
        # --- Head Pose Estimation Logic ---
        
        # 1. Define the 3D model points of a generic face.
        # These points are approximations and don't need to be exact.
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # 2. Select the corresponding 2D landmarks from the detected face.
        # The landmark indices are specific to the MediaPipe Face Mesh model.
        image_points = np.array([
            (face_landmarks.landmark[1].x * img_w, face_landmarks.landmark[1].y * img_h),     # Nose tip
            (face_landmarks.landmark[152].x * img_w, face_landmarks.landmark[152].y * img_h),   # Chin
            (face_landmarks.landmark[263].x * img_w, face_landmarks.landmark[263].y * img_h),   # Left eye left corner
            (face_landmarks.landmark[33].x * img_w, face_landmarks.landmark[33].y * img_h),    # Right eye right corner
            (face_landmarks.landmark[287].x * img_w, face_landmarks.landmark[287].y * img_h),   # Left Mouth corner
            (face_landmarks.landmark[57].x * img_w, face_landmarks.landmark[57].y * img_h)     # Right mouth corner
        ], dtype="double")

        # 3. Set up the camera matrix.
        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                               [0, focal_length, img_w / 2],
                               [0, 0, 1]])

        # 4. Use solvePnP to find the rotation and translation vectors.
        dist_coeffs = np.zeros((4, 1), dtype=np.float64) # Assuming no lens distortion
        success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs)

        # 5. Convert the rotation vector to a rotation matrix.
        rot_mat, _ = cv2.Rodrigues(rot_vec)

        # 6. Decompose the rotation matrix to get Euler angles.
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.decomposeProjectionMatrix(np.hstack((rot_mat, trans_vec)))
        
        # The angles are in degrees. We are interested in yaw and pitch.
        yaw = angles[1]
        pitch = angles[0]

        # Print the angles to the console.
        print(f"Yaw: {int(yaw)}   Pitch: {int(pitch)}")
        
        # --- End of Head Pose Estimation Logic ---

        # Draw the face mesh landmarks on the frame.
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

    cv2.imshow('INVADER', frame)


    # If the key pressed is 'q', break out of the loop.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# After the loop ends, release the webcam.
cap.release()
# Close all the windows created by OpenCV.
cv2.destroyAllWindows()
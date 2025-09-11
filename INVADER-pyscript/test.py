import cv2

# --- IMPORTANT ---
# Set this to the camera number that worked before (0 or 1)
CAMERA_ID = 0 
# ---

print(f"Attempting to open camera with ID: {CAMERA_ID}")
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print(f"Error: Cannot open camera with ID {CAMERA_ID}")
else:
    print("Camera opened successfully! Displaying feed...")
    print("Press 'q' to quit.")
    
    while True:
        # Read a frame from the camera
        success, frame = cap.read()
        
        # If we can't get a frame, something is wrong.
        if not success:
            print("Error: Failed to grab a frame from the camera.")
            break 
            
        # If we got a frame, display it
        cv2.imshow("Minimal Camera Test", frame)
        
        # Wait for the 'q' key to be pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed, quitting.")
            break
            
print("Releasing camera and closing windows.")
cap.release()
cv2.destroyAllWindows()
import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QLineEdit,
                             QPushButton, QVBoxLayout, QHBoxLayout)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWidgets import QSystemTrayIcon, QMenu
from win10toast import ToastNotifier



class InvaderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("INVADER")
        self.setGeometry(100, 100, 450, 550)
        self.toaster = ToastNotifier()

        # --- Constants ---
        self.YAW_THRESHOLD = 25
        self.PITCH_THRESHOLD = 20
        self.UNFOCUSED_TIME_LIMIT = 1
        
        # Get the absolute path to the directory where the script is running
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Join the directory path with the icon file name
        self.png_icon_path = os.path.join(script_dir, "icon.png") # For the window
        self.ico_icon_path = os.path.join(script_dir, "icon.ico") # For notifications

        # --- MediaPipe Initialization ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

        # --- Pomodoro State Variables ---
        self.session_state = "WORK"
        self.is_paused = True
        self.remaining_seconds = 0
        self.unfocused_start_time = None
        self.last_time = None

        # --- UI Setup ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        self.video_label = QLabel("Click 'Start' to begin.")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setFixedSize(400, 300)

        work_layout = QHBoxLayout()
        work_label = QLabel("Work Minutes:")
        self.work_input = QLineEdit("1")
        work_layout.addWidget(work_label)
        work_layout.addWidget(self.work_input)

        break_layout = QHBoxLayout()
        break_label = QLabel("Break Minutes:")
        self.break_input = QLineEdit("5")
        break_layout.addWidget(break_label)
        break_layout.addWidget(self.break_input)

        self.timer_label = QLabel("--:--")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer_label.setFont(QFont("Arial", 60, QFont.Weight.Bold))

        self.start_button = QPushButton("Start Pomodoro")
        self.start_button.setFont(QFont("Arial", 14))

        self.main_layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addLayout(work_layout)
        self.main_layout.addLayout(break_layout)
        self.main_layout.addWidget(self.timer_label)
        self.main_layout.addWidget(self.start_button)
        
        # --- Camera and Timer ---
        self.cap = None
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_frame)
        self.start_button.clicked.connect(self.start_pomodoro)
        
        # --- New: System Tray Icon Setup ---
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon(self.png_icon_path))
        
        tray_menu = QMenu()
        show_action = QAction("Show", self)
        quit_action = QAction("Quit", self)
        show_action.triggered.connect(self.show)
        quit_action.triggered.connect(QApplication.instance().quit)
        tray_menu.addAction(show_action)
        tray_menu.addAction(quit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def start_pomodoro(self):
        if not self.video_timer.isActive():
            try:
                self.work_minutes = int(self.work_input.text())
                self.break_minutes = int(self.break_input.text())
            except ValueError:
                self.timer_label.setText("Invalid Time!")
                return
            
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.video_label.setText("Error: Could not open camera.")
                return

            self.session_state = "WORK"
            self.remaining_seconds = self.work_minutes * 60
            self.is_paused = False
            self.last_time = time.time()
            self.video_timer.start(30)
            self.start_button.setText("Stop Pomodoro")
        else:
            self.video_timer.stop()
            if self.cap:
                self.cap.release()
            self.cap = None
            self.is_paused = True
            self.video_label.setText("Click 'Start' to begin.")
            self.timer_label.setText("--:--")
            self.start_button.setText("Start Pomodoro")

    def update_frame(self):
        success, frame = self.cap.read()
        if not success:
            return

        frame = cv2.flip(frame, 1)
        # Add this line here
        h, w, c = frame.shape

        # --- Pomodoro and CV Logic (from our old script) ---
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time

        if not self.is_paused:
            self.remaining_seconds -= delta_time

        if self.remaining_seconds <= 0:
            if self.session_state == "WORK":
                self.session_state = "BREAK"
                self.remaining_seconds = self.break_minutes * 60

               
               # New win10toast notification
                self.toaster.show_toast(
                    "Time for a Break!",
                    f"Your {self.work_minutes}-minute work session is over.",
                    icon_path=self.ico_icon_path,
                    duration=10,
                    threaded=True # Use threaded=True to not block the GUI
                )

            else: # Was "BREAK"
                self.session_state = "WORK"
                self.remaining_seconds = self.work_minutes * 60
                
                # New win10toast notification
                self.toaster.show_toast(
                    "Back to Work!",
                    f"Your {self.break_minutes}-minute break is over. Time to focus!",
                    icon_path=self.ico_icon_path,
                    duration=10,
                    threaded=True # Use threaded=True to not block the GUI
                )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        status = "UNFOCUSED"
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            # (Head pose estimation logic)
            model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])
            image_points = np.array([(face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h), (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h), (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h), (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h), (face_landmarks.landmark[287].x * w, face_landmarks.landmark[287].y * h), (face_landmarks.landmark[57].x * w, face_landmarks.landmark[57].y * h)], dtype="double")
            
            cam_matrix = np.array([[float(w), 0, h / 2], [0, float(w), w / 2], [0, 0, 1]])
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs)
            
            if success:
                rot_mat, _ = cv2.Rodrigues(rot_vec)
                _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(np.hstack((rot_mat, trans_vec)))
                yaw = angles[1]
                pitch = angles[0]
                if abs(yaw) < self.YAW_THRESHOLD and abs(pitch) < self.PITCH_THRESHOLD:
                    status = "FOCUSED"

            self.mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=self.mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=self.drawing_spec, connection_drawing_spec=self.drawing_spec)

         # --- ADD THE LOGIC HERE ---
        if self.session_state == "WORK":
            if status == "FOCUSED":
                # If we were in an unfocused state, it's safe to auto-resume
                if self.unfocused_start_time is not None:
                    self.is_paused = False
                self.unfocused_start_time = None
            else: # UNFOCUSED
                # Only start the auto-pause grace period if the timer is not already paused
                if not self.is_paused:
                    if self.unfocused_start_time is None:
                        self.unfocused_start_time = time.time()
                    elif time.time() - self.unfocused_start_time > self.UNFOCUSED_TIME_LIMIT:
                        self.is_paused = True
        else: # During a BREAK, we don't auto-pause
            self.unfocused_start_time = None

     

        # --- Draw Status on frame and Update GUI ---
        cv2.putText(frame, f"STATUS: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if status == "FOCUSED" else (0, 0, 255), 2)
        cv2.putText(frame, f"SESSION: {self.session_state}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if self.is_paused:
            cv2.putText(frame, "PAUSED", (w // 2 - 50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # Update GUI timer label
        mins = int(self.remaining_seconds // 60)
        secs = int(self.remaining_seconds % 60)
        self.timer_label.setText(f"{mins:02d}:{secs:02d}")

        # --- Convert frame to QPixmap and display ---
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled_image = qt_image.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(scaled_image))

    def closeEvent(self, event):
        # Override the window's close event
        event.ignore() # Prevent the app from quitting
        self.hide()    # Hide the main window
        self.tray_icon.showMessage(
            "INVADER is running",
            "The application is still running in the background.",
            QSystemTrayIcon.MessageIcon.Information,
            2000 # Message duration in milliseconds
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InvaderApp()
    window.show()
    sys.exit(app.exec())
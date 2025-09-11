import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import threading
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from win10toast import ToastNotifier
from pystray import Icon, MenuItem as item, Menu
from PIL import Image as PILImage


class InvaderApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("INVADER")
        self.geometry("500x650")
        self.toaster = ToastNotifier()

        # --- Constants ---
        self.YAW_THRESHOLD = 25
        self.PITCH_THRESHOLD = 20
        self.UNFOCUSED_TIME_LIMIT = 1

        # Paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.png_icon_path = os.path.join(script_dir, "icon.png")
        self.ico_icon_path = os.path.join(script_dir, "icon.ico")

        # --- MediaPipe ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1,
                                                    min_detection_confidence=0.5,
                                                    min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=1, circle_radius=1, color=(0, 255, 0))

        # --- Pomodoro State ---
        self.session_state = "WORK"
        self.is_paused = True
        self.remaining_seconds = 0
        self.unfocused_start_time = None
        self.last_time = None

        # --- GUI Setup ---
        self.video_label = ctk.CTkLabel(self, text="Click 'Start' to begin.",
                                        width=400, height=300,
                                        corner_radius=10,
                                        fg_color="black",
                                        text_color="white")
        self.video_label.pack(pady=10)

        # Work input
        work_frame = ctk.CTkFrame(self)
        work_frame.pack(pady=5)
        ctk.CTkLabel(work_frame, text="Work Minutes:").pack(side="left", padx=5)
        self.work_input = ctk.CTkEntry(work_frame, width=60)
        self.work_input.insert(0, "1")
        self.work_input.pack(side="left")

        # Break input
        break_frame = ctk.CTkFrame(self)
        break_frame.pack(pady=5)
        ctk.CTkLabel(break_frame, text="Break Minutes:").pack(side="left", padx=5)
        self.break_input = ctk.CTkEntry(break_frame, width=60)
        self.break_input.insert(0, "5")
        self.break_input.pack(side="left")

        # Timer label
        self.timer_label = ctk.CTkLabel(self, text="--:--", font=("Arial", 60, "bold"))
        self.timer_label.pack(pady=20)

        # Start button
        self.start_button = ctk.CTkButton(self, text="Start Pomodoro",
                                          command=self.start_pomodoro)
        self.start_button.pack(pady=10)

        # Camera
        self.cap = None
        self.update_job = None

        # --- System Tray Icon ---
        tray_icon_image = PILImage.open(self.png_icon_path)
        self.tray_icon = Icon("INVADER", tray_icon_image,
                              menu=Menu(
                                  item("Show", self.show_window),
                                  item("Quit", self.quit_app)
                              ))

        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def show_window(self):
        self.after(0, self.deiconify)

    def quit_app(self):
        if self.cap:
            self.cap.release()
        self.tray_icon.stop()
        self.destroy()
        sys.exit(0)

    def start_pomodoro(self):
        if self.update_job is None:
            try:
                self.work_minutes = int(self.work_input.get())
                self.break_minutes = int(self.break_input.get())
            except ValueError:
                self.timer_label.configure(text="Invalid Time!")
                return

            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.video_label.configure(text="Error: Could not open camera.")
                return

            self.session_state = "WORK"
            self.remaining_seconds = self.work_minutes * 60
            self.is_paused = False
            self.last_time = time.time()
            self.start_button.configure(text="Stop Pomodoro")
            self.update_frame()
        else:
            self.stop_pomodoro()

    def stop_pomodoro(self):
        if self.update_job:
            self.after_cancel(self.update_job)
            self.update_job = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_paused = True
        self.video_label.configure(text="Click 'Start' to begin.")
        self.timer_label.configure(text="--:--")
        self.start_button.configure(text="Start Pomodoro")

    def update_frame(self):
        success, frame = self.cap.read()
        if not success:
            self.update_job = self.after(30, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Pomodoro logic
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time

        if not self.is_paused:
            self.remaining_seconds -= delta_time

        if self.remaining_seconds <= 0:
            if self.session_state == "WORK":
                self.session_state = "BREAK"
                self.remaining_seconds = self.break_minutes * 60
                self.toaster.show_toast("Time for a Break!",
                                        f"Your {self.work_minutes}-minute work session is over.",
                                        icon_path=self.ico_icon_path, duration=10, threaded=True)
            else:
                self.session_state = "WORK"
                self.remaining_seconds = self.work_minutes * 60
                self.toaster.show_toast("Back to Work!",
                                        f"Your {self.break_minutes}-minute break is over. Time to focus!",
                                        icon_path=self.ico_icon_path, duration=10, threaded=True)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        status = "UNFOCUSED"
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0)
            ])
            image_points = np.array([
                (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),
                (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h),
                (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h),
                (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),
                (face_landmarks.landmark[287].x * w, face_landmarks.landmark[287].y * h),
                (face_landmarks.landmark[57].x * w, face_landmarks.landmark[57].y * h)
            ], dtype="double")

            cam_matrix = np.array([
                [float(w), 0, h / 2],
                [0, float(w), w / 2],
                [0, 0, 1]
            ])
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points,
                                                       cam_matrix, dist_coeffs)

            if success:
                rot_mat, _ = cv2.Rodrigues(rot_vec)
                _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(
                    np.hstack((rot_mat, trans_vec)))
                yaw = angles[1]
                pitch = angles[0]
                if abs(yaw) < self.YAW_THRESHOLD and abs(pitch) < self.PITCH_THRESHOLD:
                    status = "FOCUSED"

            self.mp_drawing.draw_landmarks(frame, face_landmarks,
                                           self.mp_face_mesh.FACEMESH_TESSELATION,
                                           self.drawing_spec, self.drawing_spec)

        # Pause logic
        if self.session_state == "WORK":
            if status == "FOCUSED":
                if self.unfocused_start_time is not None:
                    self.is_paused = False
                self.unfocused_start_time = None
            else:
                if not self.is_paused:
                    if self.unfocused_start_time is None:
                        self.unfocused_start_time = time.time()
                    elif time.time() - self.unfocused_start_time > self.UNFOCUSED_TIME_LIMIT:
                        self.is_paused = True
        else:
            self.unfocused_start_time = None

        # Overlay
        cv2.putText(frame, f"STATUS: {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if status == "FOCUSED" else (0, 0, 255), 2)
        cv2.putText(frame, f"SESSION: {self.session_state}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if self.is_paused:
            cv2.putText(frame, "PAUSED", (w // 2 - 50, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # Timer label
        mins = int(self.remaining_seconds // 60)
        secs = int(self.remaining_seconds % 60)
        self.timer_label.configure(text=f"{mins:02d}:{secs:02d}")

        # Display frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img.resize((400, 300)))
        self.video_label.configure(image=imgtk, text="")
        self.video_label.image = imgtk

        # Loop
        self.update_job = self.after(30, self.update_frame)

    def withdraw_to_tray(self, event=None):
        self.withdraw()
        self.toaster.show_toast("INVADER is running",
                                "The application is still running in the background.",
                                icon_path=self.ico_icon_path, duration=5, threaded=True)


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")  # "light" or "dark"
    ctk.set_default_color_theme("blue")

    app = InvaderApp()
    app.protocol("WM_DELETE_WINDOW", app.withdraw_to_tray)
    app.mainloop()

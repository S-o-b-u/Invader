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

        import traceback

        try:
            # Paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.png_icon_path = os.path.join(script_dir, "icon.png")
            self.ico_icon_path = os.path.join(script_dir, "icon.ico")

            # Window
            self.title("INVADER")
            self.geometry("900x600")

            # Try to set icon
            try:
                if os.path.exists(self.ico_icon_path):
                    self.iconbitmap(self.ico_icon_path)
                elif os.path.exists(self.png_icon_path):
                    self.iconphoto(False, tk.PhotoImage(file=self.png_icon_path))
            except Exception as e:
                print("⚠️ Icon load failed:", e)

            # Toast notifier (Windows-only)
            try:
                from win10toast import ToastNotifier
                self.toaster = ToastNotifier()
            except Exception as e:
                print("⚠️ ToastNotifier not available:", e)
                self.toaster = None

            # --- Tab System ---
            self.tabs = ctk.CTkTabview(self, width=480, height=650)
            self.tabs.pack(pady=10, padx=10, fill="both", expand=True)

            # Tabs
            self.tabs.add("Pomodoro")
            self.tabs.add("Dashboard")
            self.pomodoro_tab = self.tabs.tab("Pomodoro")
            self.dashboard_tab = self.tabs.tab("Dashboard")

            # Build UIs
            self.create_pomodoro_tab()
            self.create_dashboard_tab()

            # --- Constants ---
            self.YAW_THRESHOLD = 25
            self.PITCH_THRESHOLD = 20
            self.UNFOCUSED_TIME_LIMIT = 1

            # --- MediaPipe ---
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.drawing_spec = self.mp_drawing.DrawingSpec(
                thickness=1, circle_radius=1, color=(0, 255, 0)
            )

            # --- State ---
            self.session_state = "WORK"
            self.is_paused = True
            self.remaining_seconds = 0.0
            self.unfocused_start_time = None
            self.last_time = None

            # Logging
            self.logs = []
            self.focused_count = 0
            self.unfocused_count = 0

            # Camera
            self.cap = None
            self.update_job = None

            # --- System Tray ---
            try:
                from pystray import Icon, Menu, MenuItem as item
                from PIL import Image as PILImage

                if os.path.exists(self.png_icon_path):
                    tray_icon_image = PILImage.open(self.png_icon_path)
                elif os.path.exists(self.ico_icon_path):
                    tray_icon_image = PILImage.open(self.ico_icon_path)
                else:
                    tray_icon_image = PILImage.new('RGBA', (64, 64), (0, 0, 0, 0))

                self.tray_icon = Icon(
                    "INVADER", tray_icon_image,
                    menu=Menu(item("Show", self.show_window),
                            item("Quit", self.quit_app))
                )
                threading.Thread(target=self.tray_icon.run, daemon=True).start()
            except Exception as e:
                print("⚠️ Tray icon failed:", e)
                self.tray_icon = None

        except Exception as e:
            print("💥 Crash in InvaderApp init:", e)
            traceback.print_exc()

    # ----------------- UI Builders -----------------
    def create_pomodoro_tab(self):
        self.pomodoro_tab.grid_columnconfigure(0, weight=2)
        self.pomodoro_tab.grid_columnconfigure(1, weight=1)
        self.pomodoro_tab.grid_rowconfigure(0, weight=1)

        # Left = Camera
        self.camera_frame = ctk.CTkFrame(self.pomodoro_tab, corner_radius=10)
        self.camera_frame.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")

        ctk.CTkLabel(
            self.camera_frame, text="📷 Live Camera", font=("Arial", 16, "bold")
        ).pack(pady=(8, 5))

        self.camera_label = ctk.CTkLabel(self.camera_frame, text="Camera feed here")
        self.camera_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Right = Controls
        self.controls_frame = ctk.CTkFrame(self.pomodoro_tab, corner_radius=10)
        self.controls_frame.grid(row=0, column=1, padx=8, pady=8, sticky="nsew")

        # Inputs
        ctk.CTkLabel(self.controls_frame, text="Work (minutes):").pack(pady=(10, 0))
        self.work_input = ctk.CTkEntry(self.controls_frame)
        self.work_input.insert(0, "25")
        self.work_input.pack(pady=(0, 10), padx=20, fill="x")

        ctk.CTkLabel(self.controls_frame, text="Break (minutes):").pack(pady=(10, 0))
        self.break_input = ctk.CTkEntry(self.controls_frame)
        self.break_input.insert(0, "5")
        self.break_input.pack(pady=(0, 10), padx=20, fill="x")

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.controls_frame)
        self.progress_bar.pack(pady=10, padx=20, fill="x")
        self.progress_bar.set(0)

        # Controls
        ctk.CTkLabel(
            self.controls_frame, text="⏱️ Pomodoro Controls", font=("Arial", 16, "bold")
        ).pack(pady=(8, 5))

        self.timer_label = ctk.CTkLabel(
            self.controls_frame, text="25:00", font=("Arial", 32, "bold")
        )
        self.timer_label.pack(pady=(10, 20))

        self.start_button = ctk.CTkButton(
            self.controls_frame, text="▶️ Start", command=self.start_pomodoro
        )
        self.start_button.pack(pady=5, fill="x", padx=20)

        self.pause_button = ctk.CTkButton(
            self.controls_frame, text="⏸️ Pause", command=self.manual_toggle_pause
        )
        self.pause_button.pack(pady=5, fill="x", padx=20)

        self.reset_button = ctk.CTkButton(
            self.controls_frame, text="🔄 Reset", command=self.stop_pomodoro
        )
        self.reset_button.pack(pady=5, fill="x", padx=20)

        self.status_label = ctk.CTkLabel(self.controls_frame, text="Status: Idle")
        self.status_label.pack(pady=(20, 10))

    def create_dashboard_tab(self):
        self.dashboard_tab.grid_columnconfigure(0, weight=2)
        self.dashboard_tab.grid_columnconfigure(1, weight=1)
        self.dashboard_tab.grid_rowconfigure(0, weight=1)

        # Logs
        self.log_frame = ctk.CTkFrame(self.dashboard_tab, corner_radius=10)
        self.log_frame.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")

        ctk.CTkLabel(
            self.log_frame, text="📜 Session Logs", font=("Arial", 16, "bold")
        ).pack(pady=(10, 5))

        self.log_scroll = ctk.CTkScrollableFrame(self.log_frame, width=480, height=360)
        self.log_scroll.pack(fill="both", expand=True, padx=10, pady=5)

        self.summary_label = ctk.CTkLabel(self.log_frame, text="No stats yet.", anchor="w")
        self.summary_label.pack(pady=(6, 12), padx=10, anchor="w")

        # Stats
        self.stats_frame = ctk.CTkFrame(self.dashboard_tab, corner_radius=10)
        self.stats_frame.grid(row=0, column=1, padx=12, pady=(12, 6), sticky="new")

        ctk.CTkLabel(
            self.stats_frame, text="📊 Focus Stats", font=("Arial", 16, "bold")
        ).pack(pady=(8, 4))

        self.focus_label = ctk.CTkLabel(self.stats_frame, text="✅ Focused: 0 times")
        self.focus_label.pack(anchor="w", padx=10, pady=(6, 0))

        self.unfocus_label = ctk.CTkLabel(self.stats_frame, text="⚠️ Unfocused: 0 times")
        self.unfocus_label.pack(anchor="w", padx=10, pady=(2, 0))

        self.focus_percent_label = ctk.CTkLabel(self.stats_frame, text="📈 Focus %: 0%")
        self.focus_percent_label.pack(anchor="w", padx=10, pady=(2, 10))

        # History (dark listbox replacement)
        self.history_frame = ctk.CTkFrame(self.dashboard_tab, corner_radius=10)
        self.history_frame.grid(row=1, column=1, padx=12, pady=(6, 12), sticky="nsew")

        ctk.CTkLabel(
            self.history_frame, text="📂 History", font=("Arial", 16, "bold")
        ).pack(pady=(8, 4))

        # Dark themed Listbox
        self.history_list = tk.Listbox(
            self.history_frame,
            height=6,
            bg="gray20", fg="white",    # Match dark mode
            highlightthickness=0,
            bd=0,
            selectbackground="gray40",
            selectforeground="white"
        )
        self.history_list.pack(fill="both", expand=True, padx=10, pady=10)

    # ----------------- UI / App helpers -----------------
    def show_window(self):
        self.after(0, self.deiconify)

    def quit_app(self):
        # cleanup
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        try:
            if self.tray_icon:
                self.tray_icon.stop()
        except Exception:
            pass
        self.destroy()
        sys.exit(0)

    def manual_toggle_pause(self):
        # manual pause/resume from UI
        self.is_paused = not self.is_paused
        self._log_event(f"MANUAL: {'Paused' if self.is_paused else 'Resumed'}")
        # update immediately so UI reflects state
        self.update_timer_label_display()

    # ----------------- Logging & Dashboard -----------------
    def _log_event(self, message: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        self.logs.append(line)

        # Insert into scrollable frame (label per line)
        log_label = ctk.CTkLabel(self.log_scroll, text=line, anchor="w")
        log_label.pack(fill="x", padx=5, pady=2)

        self.update_summary()


    def update_summary(self):
        # A simple summary: number of log lines, last events
        total = len(self.logs)
        last = self.logs[-1] if total else "No events yet."
        summary = f"Total events: {total}    Last: {last}"
        self.summary_label.configure(text=summary)

    # ----------------- Pomodoro & Camera -----------------
    def start_pomodoro(self):
        if self.update_job is None:
            # start
            try:
                self.work_minutes = int(self.work_input.get())
                self.break_minutes = int(self.break_input.get())
            except ValueError:
                self.timer_label.configure(text="Invalid Time!")
                return

            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.camera_label.configure(text="Error: Could not open camera.")
                return

            self.session_state = "WORK"
            self.remaining_seconds = float(self.work_minutes * 60)
            self.is_paused = False
            self.last_time = time.time()
            self.start_button.configure(text="Stop Pomodoro")
            self._log_event(f"Started Pomodoro: work={self.work_minutes}m break={self.break_minutes}m")
            self.update_frame()
        else:
            self.stop_pomodoro()

    def stop_pomodoro(self):
        if self.update_job:
            self.after_cancel(self.update_job)
            self.update_job = None
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_paused = True
        self.camera_label.configure(text="Click 'Start' to begin.", image=None)
        self.timer_label.configure(text="--:--")
        self.start_button.configure(text="Start Pomodoro")
        self._log_event("Stopped Pomodoro")

    def update_timer_label_display(self):
        mins = int(max(0, self.remaining_seconds) // 60)
        secs = int(max(0, self.remaining_seconds) % 60)
        self.timer_label.configure(text=f"{mins:02d}:{secs:02d}")

        # Update progress bar
        if self.session_state == "WORK":
            total = self.work_minutes * 60
        else:
            total = self.break_minutes * 60
        # avoid division by zero
        if total <= 0:
            progress = 0
        else:
            progress = max(0, min(1, (total - self.remaining_seconds) / total))
        self.progress_bar.set(progress)

    def update_frame(self):
        # Read frame
        success, frame = self.cap.read()
        if not success:
            # schedule next attempt
            self.update_job = self.after(30, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Pomodoro time update
        current_time = time.time()
        delta_time = current_time - self.last_time if self.last_time is not None else 0
        self.last_time = current_time

        if not self.is_paused:
            self.remaining_seconds -= delta_time

        # Session end handling
        if self.remaining_seconds <= 0:
            if self.session_state == "WORK":
                # move to break
                self.session_state = "BREAK"
                self.remaining_seconds = float(self.break_minutes * 60)
                self.toaster.show_toast("Time for a Break!",
                                        f"Your {self.work_minutes}-minute work session is over.",
                                        icon_path=self.ico_icon_path if os.path.exists(self.ico_icon_path) else None,
                                        duration=7, threaded=True)
                self._log_event(f"WORK -> BREAK (work {self.work_minutes}m finished)")

                # Add session summary to history here (work session finished)
                focus_percent = (self.focused_count / max(1, self.focused_count + self.unfocused_count)) * 100
                summary = f"Work Session: {self.work_minutes}m | Focus: {focus_percent:.1f}%"
                self.history_list.insert("end", summary)

                # Reset counters for next session
                self.focused_count = 0
                self.unfocused_count = 0
            else:
                # move to work
                self.session_state = "WORK"
                self.remaining_seconds = float(self.work_minutes * 60)
                self.toaster.show_toast("Back to Work!",
                                        f"Your {self.break_minutes}-minute break is over. Time to focus!",
                                        icon_path=self.ico_icon_path if os.path.exists(self.ico_icon_path) else None,
                                        duration=7, threaded=True)
                self._log_event(f"BREAK -> WORK (break {self.break_minutes}m finished)")

        # Face detection & head-pose estimation
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        status = "UNFOCUSED"
        if results and results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            try:
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
                ok, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs)

                if ok:
                    rot_mat, _ = cv2.Rodrigues(rot_vec)
                    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(
                        np.hstack((rot_mat, trans_vec)))
                    yaw = angles[1]
                    pitch = angles[0]
                    if abs(yaw) < self.YAW_THRESHOLD and abs(pitch) < self.PITCH_THRESHOLD:
                        status = "FOCUSED"
            except Exception:
                # If headpose math fails, keep UNFOCUSED status
                pass

            # draw landmarks for visualization
            try:
                self.mp_drawing.draw_landmarks(frame, face_landmarks,
                                               self.mp_face_mesh.FACEMESH_TESSELATION,
                                               self.drawing_spec, self.drawing_spec)
            except Exception:
                pass

        # Auto-pause logic only during WORK
        if self.session_state == "WORK":
            if status == "FOCUSED":
                # if previously flagged as unfocused and now regained focus -> resume automatically
                if self.unfocused_start_time is not None and self.is_paused:
                    self.is_paused = False
                    self._log_event("AUTO: Resumed (regained focus)")
                self.unfocused_start_time = None
                self.focused_count += 1
            else:
                # UNFOCUSED
                self.unfocused_count += 1
                if not self.is_paused:
                    if self.unfocused_start_time is None:
                        self.unfocused_start_time = time.time()
                    elif time.time() - self.unfocused_start_time > self.UNFOCUSED_TIME_LIMIT:
                        self.is_paused = True
                        self._log_event("AUTO: Paused (unfocused)")
        else:
            # during break, clear unfocused timer
            self.unfocused_start_time = None

        # Draw overlay text on frame
        cv2.putText(frame, f"STATUS: {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if status == "FOCUSED" else (0, 0, 255), 2)
        cv2.putText(frame, f"SESSION: {self.session_state}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if self.is_paused:
            cv2.putText(frame, "PAUSED", (w // 2 - 50, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # Update timer label
        self.update_timer_label_display()

        # Convert to image and display in CTk label
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img.resize((400, 300)))
        self.camera_label.configure(image=imgtk, text="")
        self.camera_label.image = imgtk

        # schedule next frame
        self.update_job = self.after(30, self.update_frame)

        # Update stats labels
        total_events = self.focused_count + self.unfocused_count
        if total_events > 0:
            focus_percent = (self.focused_count / total_events) * 100
        else:
            focus_percent = 0
        self.focus_label.configure(text=f"Focused: {self.focused_count}")
        self.unfocus_label.configure(text=f"Unfocused: {self.unfocused_count}")
        self.focus_percent_label.configure(text=f"Focus %: {focus_percent:.1f}%")

    def withdraw_to_tray(self, event=None):
        # minimize to tray instead of closing
        self.withdraw()
        try:
            # show a toast to notify user
            self.toaster.show_toast("INVADER is running",
                                    "The application is still running in the background.",
                                    icon_path=self.ico_icon_path if os.path.exists(self.ico_icon_path) else None,
                                    duration=5, threaded=True)
        except Exception:
            pass


if __name__ == "__main__":
    print("Launching Invader...")
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    app = InvaderApp()
    print("App created")
    app.protocol("WM_DELETE_WINDOW", app.withdraw_to_tray)
    app.mainloop()
    print("Exited mainloop")


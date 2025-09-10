# Invader 👾
**Invade your distractions. Reclaim your focus.**

Invader is a smart Pomodoro timer built with Python that uses computer vision to track your attention. It's designed for those deep work sessions where every ounce of focus counts. If you look away from the screen, the timer automatically pauses, holding you accountable and keeping you in the zone.

> *Currently Building*

---

## ✨ Core Features
- 🧠 **Smart Attention Tracking**: Uses your webcam and MediaPipe's face mesh to detect your head pose. If you look away, the timer pauses. Look back, and it resumes.  
- 🍅 **Automatic Pomodoro Cycles**: Runs on the classic Pomodoro Technique (25 min work / 5 min break) to prevent burnout.  
- 🖥️ **Real-time Visual Feedback**: All information is overlaid directly onto your webcam feed, so you never have to leave your window.  
- ⌨️ **Simple Controls**: Use your keyboard to manually pause/resume or quit the application.  

---

## 🔧 How It Works
Invader uses the **OpenCV** library to capture your webcam feed and **MediaPipe** to create a 3D map of your face in real-time.

By analyzing the rotation of your head (specifically the yaw and pitch), the script can determine if you are looking forward at the screen or if your attention has drifted. A few seconds of being "unfocused" will pause the timer, creating a gentle but effective feedback loop to keep you on task.

---

## 🚀 Getting Started

### Prerequisites
- Python **3.7+** installed on your system.

### Installation
Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/invader-focus-timer.git
cd invader-focus-timer
````

Create a virtual environment (recommended):

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python invader.py
```

Your webcam will turn on, and the application will start in a **WORK** session.

---

## 🕹️ How to Use

* Look at the screen to keep the timer running.
* Look away for more than 2 seconds, and the timer will pause.
* Press **SPACEBAR** to manually pause or resume the timer.
* Press **q** to quit the application.

---

## 🛣️ Future Roadmap

This Python version is the foundation. The ultimate goal is to make **Invader** a cross-platform tool.

* [ ] **Browser Extension Version**: Port the logic to JavaScript using MediaPipe for Web.
* [ ] **Sound Alerts**: Optional notifications for pause/resume/session end.
* [ ] **Focus Analytics**: Log sessions to track productivity over time.
* [ ] **Customizable Timers**: User-defined work/break durations.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.


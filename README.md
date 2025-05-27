# Real-Time Human Body and Facial Emotion Detection with MediaPipe

This Python script utilizes the **MediaPipe Holistic** model to perform real-time human body, face, and hand landmark detection. It also includes a basic rule-based system to infer facial emotions such as **Happiness**, **Surprise**, **Anger**, **Sadness**, and **Neutral**.

---

## 📌 Features

* ✅ Real-time webcam processing
* ✅ Full body pose detection
* ✅ Face mesh detection (eyes, mouth, eyebrows)
* ✅ Hand skeleton tracking (left and right)
* ✅ Simple facial emotion inference
* ✅ Visual overlay of landmarks and detected emotion

---

## 🧠 How It Works

The script uses **MediaPipe's Holistic** solution to detect:

* **Pose landmarks** across the full body
* **Face landmarks** including eyes, eyebrows, and mouth
* **Hand landmarks** for both hands

Then, based on geometric relationships between facial landmarks (like distances between eyebrows, mouth corners, and eyes), it infers basic facial emotions using simple threshold-based logic.

This makes it an excellent **starting point** for learning about real-time facial analysis and emotion recognition.

---

## ⚙️ Prerequisites

* Python 3.x
* A webcam connected to your machine

---

## 🔧 Installation

Open a terminal and run the following:

```bash
pip install opencv-python mediapipe
```

---

## ▶️ Running the Script

1. **Save the code** as a Python file, e.g. `emotion_detector.py`
2. **Run the script**:

```bash
python emotion_detector.py
```

3. A window titled **"Human Body and Face Landmark Detection"** will open.
4. Your webcam feed will display:

   * Pose, face, and hand landmarks
   * The current detected facial emotion
5. Press **`q`** to exit the application.

---

## 📁 Repository Structure (Suggested)

```
emotion-detector/
├── emotion_detector.py
├── README.md
└── requirements.txt (optional)
```

---

## 📌 Notes

* Emotion detection is **basic** and based on heuristic geometry rules.
* This script is ideal for **educational and experimental** purposes.
* For better accuracy and robustness, consider using machine learning-based emotion classifiers.

---

## 🙌 Contributions

Feel free to open issues, suggest improvements, or submit pull requests to enhance:

* Emotion detection logic
* GUI improvements
* Additional features (e.g., logging, emotion history, alerts)

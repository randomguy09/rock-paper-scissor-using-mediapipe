import os
import time
import random
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model (first run only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
hand_options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2


def classify_gesture(landmarks, handedness_label):
    def finger_extended(tip_idx, pip_idx):
        return landmarks[tip_idx][1] < landmarks[pip_idx][1]

    index_ext = finger_extended(8, 6)
    middle_ext = finger_extended(12, 10)
    ring_ext = finger_extended(16, 14)
    pinky_ext = finger_extended(20, 18)

    if handedness_label == "Right":
        thumb_ext = landmarks[THUMB_TIP][0] < landmarks[THUMB_MCP][0]
    else:
        thumb_ext = landmarks[THUMB_TIP][0] > landmarks[THUMB_MCP][0]

    extended = [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]
    extended_count = sum(extended)

    if extended_count <= 1 and not (index_ext and middle_ext):
        return "rock"

    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "scissors"

    if extended_count >= 4:
        return "paper"

    return "unknown"


class StabilityTracker:
    REQUIRED_STABLE_FRAMES = 8

    def __init__(self):
        self.last_label = "unknown"
        self.streak = 0

    def update(self, label):
        if label == self.last_label:
            self.streak += 1
        else:
            self.last_label = label
            self.streak = 1
        return self.is_stable()

    def is_stable(self):
        return self.last_label != "unknown" and self.streak >= self.REQUIRED_STABLE_FRAMES

    @property
    def stable_label(self):
        return self.last_label if self.is_stable() else None

    def reset(self):
        self.last_label = "unknown"
        self.streak = 0


class GameStateMachine:
    WAITING = "WAITING"
    COUNTDOWN = "COUNTDOWN"
    REVEAL = "REVEAL"
    RESULT = "RESULT"

    COUNTDOWN_SECONDS = 3
    RESULT_DISPLAY_SECONDS = 2.5

    CHOICES = ["rock", "paper", "scissors"]
    BEATS = {
        "rock": "scissors",
        "scissors": "paper",
        "paper": "rock",
    }

    def __init__(self):
        self.state = self.WAITING
        self.state_start_time = time.time()
        self.computer_choice = None
        self.player_choice = None
        self.result_text = ""
        self.score_player = 0
        self.score_computer = 0

    def start_round(self):
        if self.state == self.WAITING:
            self.state = self.COUNTDOWN
            self.state_start_time = time.time()
            self.player_choice = None
            self.computer_choice = None
            self.result_text = ""

    def elapsed(self):
        return time.time() - self.state_start_time

    def update(self, stable_player_label):
        if self.state == self.COUNTDOWN:
            remaining = self.COUNTDOWN_SECONDS - self.elapsed()
            if remaining <= 0:
                self.player_choice = stable_player_label or "unknown"
                self.computer_choice = random.choice(self.CHOICES)
                self._decide_winner()
                self.state = self.REVEAL
                self.state_start_time = time.time()

        elif self.state == self.REVEAL:
            if self.elapsed() >= 0.8:
                self.state = self.RESULT
                self.state_start_time = time.time()

        elif self.state == self.RESULT:
            if self.elapsed() >= self.RESULT_DISPLAY_SECONDS:
                self.state = self.WAITING
                self.state_start_time = time.time()

    def _decide_winner(self):
        p, c = self.player_choice, self.computer_choice
        if p == "unknown" or p is None:
            self.result_text = "No gesture detected - try again!"
        elif p == c:
            self.result_text = "Draw!"
        elif self.BEATS[p] == c:
            self.result_text = "You win!"
            self.score_player += 1
        else:
            self.result_text = "Computer wins!"
            self.score_computer += 1

    def reset_score(self):
        self.score_player = 0
        self.score_computer = 0


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    tracker = StabilityTracker()
    game = GameStateMachine()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        current_label = "unknown"

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            handedness_label = result.handedness[0][0].category_name

            draw_hand(frame, hand_landmarks)

            pts = [(lm.x, lm.y) for lm in hand_landmarks]
            current_label = classify_gesture(pts, handedness_label)
        else:
            tracker.reset()

        stable = tracker.update(current_label)
        game.update(tracker.stable_label if stable else None)

        draw_ui(frame, game, current_label, tracker)

        cv2.imshow("Rock Paper Scissors - MediaPipe", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            game.start_round()
        elif key == ord("r"):
            game.reset_score()

    cap.release()
    cv2.destroyAllWindows()


def draw_hand(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def draw_ui(frame, game, current_label, tracker):
    h, w, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
    cv2.putText(frame, f"You: {game.score_player}   Computer: {game.score_computer}",
                (10, 28), font, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Detecting: {current_label} ({tracker.streak}/{tracker.REQUIRED_STABLE_FRAMES})",
                (10, h - 15), font, 0.6, (200, 200, 0), 2)

    center_x = w // 2

    if game.state == GameStateMachine.WAITING:
        cv2.putText(frame, "Press SPACE to play", (center_x - 160, h // 2),
                    font, 1.0, (255, 255, 255), 2)

    elif game.state == GameStateMachine.COUNTDOWN:
        remaining = max(0, int(game.COUNTDOWN_SECONDS - game.elapsed()) + 1)
        cv2.putText(frame, str(remaining), (center_x - 20, h // 2),
                    font, 3.0, (0, 255, 255), 4)
        cv2.putText(frame, "Show your hand!", (center_x - 140, h // 2 + 60),
                    font, 0.9, (0, 255, 255), 2)

    elif game.state in (GameStateMachine.REVEAL, GameStateMachine.RESULT):
        cv2.putText(frame, f"You: {game.player_choice}", (30, 100),
                    font, 1.0, (0, 200, 255), 2)
        cv2.putText(frame, f"CPU: {game.computer_choice}", (30, 140),
                    font, 1.0, (0, 200, 255), 2)
        if game.state == GameStateMachine.RESULT:
            cv2.putText(frame, game.result_text, (center_x - 180, h // 2),
                        font, 1.2, (0, 255, 0), 3)


if __name__ == "__main__":
    main()
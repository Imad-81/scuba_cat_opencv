import cv2
import mediapipe as mp
import numpy as np
import time

# -------------------------------
# SETTINGS
# -------------------------------
WAVE_HISTORY = 30
WAVE_THRESHOLD = 300
COOLDOWN = 3
VIDEO_PATH = "scuba_cat.MP4"

# -------------------------------
# VIDEO STATE
# -------------------------------
video_cap = None
playing_video = False

# -------------------------------
# INIT
# -------------------------------
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands = mp_hands.Hands(max_num_hands=2)
face = mp_face.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(1)

wave_positions = []
last_trigger_time = 0


# -------------------------------
# HELPERS
# -------------------------------
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def detect_wave(positions):
    if len(positions) < WAVE_HISTORY:
        return False

    diffs = np.diff(positions)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    total_movement = max(positions) - min(positions)

    return sign_changes >= 2 and total_movement > WAVE_THRESHOLD


def play_video():
    global video_cap, playing_video

    if playing_video:
        return

    video_cap = cv2.VideoCapture(VIDEO_PATH)
    playing_video = True


# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb)
    face_results = face.process(rgb)

    nose = None
    nose_covered = False
    waving = False

    # -------------------------------
    # NOSE + HAND OVERLAP DETECTION
    # -------------------------------
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        nose_landmark = face_landmarks.landmark[1]

        nx = int(nose_landmark.x * w)
        ny = int(nose_landmark.y * h)
        nose = (nx, ny)

        cv2.circle(frame, nose, 5, (0, 255, 0), -1)

        NOSE_RADIUS = 75

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:

                fingertip_ids = [4, 8, 12, 16, 20]

                for tip_id in fingertip_ids:
                    tip = hand_landmarks.landmark[tip_id]
                    tx, ty = int(tip.x * w), int(tip.y * h)

                    cv2.circle(frame, (tx, ty), 5, (255, 255, 0), -1)

                    if distance((tx, ty), nose) < NOSE_RADIUS:
                        nose_covered = True
                        break

                if nose_covered:
                    break

    # -------------------------------
    # WAVE DETECTION
    # -------------------------------
    if hand_results.multi_hand_landmarks:
        hand_centers = []

        for hand_landmarks in hand_results.multi_hand_landmarks:
            xs = []
            ys = []

            for lm in hand_landmarks.landmark:
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))

            cx = int(np.mean(xs))
            cy = int(np.mean(ys))

            hand_centers.append((cx, cy))
            cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)

        if nose:
            hand_centers.sort(key=lambda p: distance(p, nose), reverse=True)

        wave_hand = hand_centers[0]
        wave_positions.append(wave_hand[0])

        if len(wave_positions) > WAVE_HISTORY:
            wave_positions.pop(0)

        waving = detect_wave(wave_positions)

    else:
        wave_positions.clear()

    # -------------------------------
    # TRIGGER
    # -------------------------------
    current_time = time.time()

    if nose_covered and waving and (current_time - last_trigger_time > COOLDOWN):
        print("SCUBA CAT TRIGGERED 🐱🌊")
        play_video()
        last_trigger_time = current_time

    # -------------------------------
    # VIDEO WINDOW (SEPARATE)
    # -------------------------------
    if playing_video:
        ret_vid, frame_vid = video_cap.read()

        if not ret_vid:
            playing_video = False
            video_cap.release()
            cv2.destroyWindow("Meme Player")
        else:
            frame_vid = cv2.resize(frame_vid, (480, 360))
            cv2.imshow("Meme Player", frame_vid)

    # -------------------------------
    # DEBUG TEXT
    # -------------------------------
    cv2.putText(frame, f"Nose Covered: {nose_covered}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Waving: {waving}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Scuba Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()
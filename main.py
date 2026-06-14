import face_recognition
import numpy as np
import cv2
import os
import re
from anti_spoofing import ActiveLivenessChallenge, analyze_frame
# =========================================================
# LOAD KNOWN FACES
# =========================================================
def load_known_faces(folder="images"):
    known_encodings = []
    known_names = []
    if not os.path.exists(folder):
        print(" Images folder not found!")
        return known_encodings, known_names
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                name = re.split(r'\d+', file)[0]
                name = os.path.splitext(name)[0]
                known_names.append(name)
        except:
            continue
    print(f" Loaded {len(known_names)} known faces")
    return known_encodings, known_names
def compare_two_images(img1_path, img2_path):
    try:
        img1 = face_recognition.load_image_file(img1_path)
        img2 = face_recognition.load_image_file(img2_path)
        enc1 = face_recognition.face_encodings(img1)
        enc2 = face_recognition.face_encodings(img2)
        if len(enc1) == 0 or len(enc2) == 0:
            print(" No face detected")
            return
        distance = np.linalg.norm(enc1[0] - enc2[0])
        print("\n========== RESULT ==========")
        print("Face Distance:", round(distance, 4))
        print(" Same Person" if distance < 0.50 else " Different Person")
    except Exception as e:
        print("ERROR:", e)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C) if C > 0 else 0

def point_center(points):
    pts = np.array(points, dtype=np.float32)
    return tuple(np.mean(pts, axis=0))

def build_liveness_landmarks(face_landmarks):
    required = ["left_eye", "right_eye", "nose_tip", "top_lip", "chin"]
    if not all(key in face_landmarks for key in required):
        return None
    top_lip = face_landmarks["top_lip"]
    chin = face_landmarks["chin"]
    if len(top_lip) < 7 or len(chin) < 9:
        return None
    return {
        "left_eye": point_center(face_landmarks["left_eye"]),
        "right_eye": point_center(face_landmarks["right_eye"]),
        "nose_tip": point_center(face_landmarks["nose_tip"]),
        "mouth_left": tuple(top_lip[0]),
        "mouth_right": tuple(top_lip[6]),
        "chin": tuple(chin[8]),
    }

def scale_landmarks(landmarks, factor=2):
    if landmarks is None:
        return None
    return {key: (value[0] * factor, value[1] * factor) for key, value in landmarks.items()}

def failed_check_names(decision):
    failed = [check.name for check in decision.checks if not check.passed]
    return ", ".join(failed[:2]) if failed else "challenge"
# =========================================================
# SMART CAMERA DETECTION
# =========================================================
def find_working_camera(max_index=10):
    print(" Searching for working camera...")
    best_index = -1
    best_score = -1
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        brightness_values = []
        for _ in range(12):
            ret, frame = cap.read()
            if ret and frame is not None:
                brightness_values.append(np.mean(frame))
        cap.release()
        if len(brightness_values) < 8:
            continue
        avg_brightness = np.mean(brightness_values)
        variance = np.var(brightness_values)
        score = avg_brightness + variance * 10
        print(f"Camera {i}: Brightness={avg_brightness:.1f}, Variance={variance:.1f}")
        if score > best_score and avg_brightness > 30:
            best_score = score
            best_index = i
    if best_index != -1:
        print(f" Using Camera Index: {best_index}")
        return best_index
    print(" No working camera found!")
    return -1
# =========================================================
# FINAL REAL-TIME SYSTEM
# =========================================================
def real_time_system():
    known_encodings, known_names = load_known_faces()
    cam_index = find_working_camera()
    if cam_index == -1:
        return
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    EAR_THRESHOLD = 0.255
    CLOSED_FRAMES_THRESHOLD = 3
    face_data = {}
    next_face_id = 0
    previous_frame = None
    print("\n Press Q or ESC to exit\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        encodings = face_recognition.face_encodings(rgb_small, face_locations)
        landmarks_list = face_recognition.face_landmarks(rgb_small, face_locations)
        static_count = 0
        for i, (top, right, bottom, left) in enumerate(face_locations):
            top, right, bottom, left = [int(x * 2) for x in (top, right, bottom, left)]
            center = ((left + right) // 2, (top + bottom) // 2)
            # Multi-face tracking
            matched_id = None
            for fid, d in face_data.items():
                if np.linalg.norm(np.array(center) - np.array(d["center"])) < 100:
                    matched_id = fid
                    break
            if matched_id is None:
                matched_id = next_face_id
                face_data[matched_id] = {
                    "center": center,
                    "blink_count": 0,
                    "closed_frames": 0,
                    "state": "OPEN",
                    "name": "Unknown",
                    "challenge": None
                }
                next_face_id += 1
            data = face_data[matched_id]
            data["center"] = center
            # Recognition
            name = "Unknown"
            if i < len(encodings) and known_encodings:
                distances = face_recognition.face_distance(known_encodings, encodings[i])
                if len(distances) > 0 and min(distances) < 0.50:
                    name = known_names[np.argmin(distances)]
            live_person = True
            eyes_visible = False
            ear = 0.0
            liveness_status = "Checking"
            if i < len(landmarks_list):
                lm = landmarks_list[i]
                if "left_eye" in lm and "right_eye" in lm:
                    left_eye = np.array(lm["left_eye"]) * 2
                    right_eye = np.array(lm["right_eye"]) * 2
                    left_w = np.linalg.norm(left_eye[0] - left_eye[3])
                    right_w = np.linalg.norm(right_eye[0] - right_eye[3])
                    eyes_visible = (left_w > 18 and right_w > 18)
                    if eyes_visible:
                        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                    liveness_landmarks = scale_landmarks(build_liveness_landmarks(lm))
                    decision = analyze_frame(
                        frame,
                        (left, top, right - left, bottom - top),
                        landmarks=liveness_landmarks,
                        previous_frame_bgr=previous_frame,
                    )
                    if data["challenge"] is None and liveness_landmarks is not None:
                        data["challenge"] = ActiveLivenessChallenge()
                        data["challenge"].start(liveness_landmarks)
                    challenge_passed = False
                    if data["challenge"] is not None and liveness_landmarks is not None:
                        challenge_passed = data["challenge"].update(liveness_landmarks)
                    live_person = decision.live and challenge_passed
                    if not live_person:
                        static_count += 1
                        if not decision.live:
                            liveness_status = "Spoof: " + failed_check_names(decision)
                        elif data["challenge"] is not None and data["challenge"].state is not None:
                            liveness_status = "Do: " + data["challenge"].state.action.value
            # ===================== BLINK DETECTION =====================
            # Reset blink state for static images
            if not live_person:
                data["state"] = "OPEN"
                data["closed_frames"] = 0
            if eyes_visible and live_person:
                if data["state"] == "OPEN":
                    if ear < EAR_THRESHOLD:
                        data["state"] = "CLOSED"
                        data["closed_frames"] = 1
                else:
                    if ear < EAR_THRESHOLD:
                        data["closed_frames"] += 1
                    else:
                        if data["closed_frames"] >= CLOSED_FRAMES_THRESHOLD:
                            data["blink_count"] += 1
                            print(f" Blink #{data['blink_count']} | {name} (ID{matched_id})")
                        data["state"] = "OPEN"
                        data["closed_frames"] = 0
                status = "Blinking" if data["state"] == "CLOSED" else "Live"
                color = (0, 0, 255) if data["state"] == "CLOSED" else (0, 255, 0)
            else:
                status = liveness_status if not live_person else "Eyes Not Visible"
                color = (255, 0, 255) if not live_person else (0, 165, 255)
            # Draw
            box_color = (0, 255, 0) if live_person else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.putText(frame, f"{name} (ID{matched_id})", (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, status, (left, bottom+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            # Show blink count ONLY for live persons
            if live_person:
                cv2.putText(frame, f"Blinks: {data['blink_count']}", (left, bottom+55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Rejected", (left, bottom+55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        real_count = len(face_locations) - static_count
        cv2.putText(frame, f"Live Persons: {real_count}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Face Recognition + Liveness", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            print(" Exiting...")
            break
        previous_frame = frame.copy()
    cap.release()
    cv2.destroyAllWindows()
def main():
    print("\n========== FACE RECOGNITION SYSTEM ==========")
    print("1. Compare Two Images")
    print("2. Real-Time Face + Liveness")
    print("============================================")
    choice = input("Enter choice: ").strip()
    if choice == "1":
        img1 = input("First image path: ").strip()
        img2 = input("Second image path: ").strip()
        compare_two_images(img1, img2)
    elif choice == "2":
        real_time_system()
    else:
        print("Invalid choice")
if __name__ == "__main__":
    main()

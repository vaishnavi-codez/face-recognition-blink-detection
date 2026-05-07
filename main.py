import face_recognition
import numpy as np
import cv2
import os
import re
# ============================================
# LOAD KNOWN FACES
# ============================================
def load_known_faces(folder="images"):
    encodings = []
    names = []
    if not os.path.exists(folder):
        print(" images folder not found")
        return encodings, names
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            img = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(img)
            if len(enc) > 0:
                encodings.append(enc[0])
                # remove numbers from filename
                name = re.split(r'\d+', file)[0]
                name = os.path.splitext(name)[0]
                names.append(name)
        except:
            continue
    print(f" Loaded {len(names)} known faces")
    return encodings, names
# ============================================
# IMAGE COMPARISON
# ============================================
def compare_faces(img1_path, img2_path):
    try:
        img1 = face_recognition.load_image_file(img1_path)
        img2 = face_recognition.load_image_file(img2_path)
        enc1 = face_recognition.face_encodings(img1)
        enc2 = face_recognition.face_encodings(img2)
        if len(enc1) == 0 or len(enc2) == 0:
            print(" Face not detected in one of the images")
            return
        enc1 = enc1[0]
        enc2 = enc2[0]
        distance = np.linalg.norm(enc1 - enc2)
        print("\n========== RESULT ==========")
        print("Distance:", round(distance, 4))
        if distance < 0.5:
            print(" Same Person")
        else:
            print(" Different Person")
    except Exception as e:
        print("Error:", e)
# ============================================
# EAR FUNCTION
# ============================================
def EAR(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    if C == 0:
        return 0
    return (A + B) / (2.0 * C)
# ============================================
# AUTO CAMERA DETECTION
# ============================================
def find_camera():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            cap.release()
            print(f" Using camera index: {i}")
            return i
        cap.release()
    return -1
# ============================================
# FACE + BLINK SYSTEM
# ============================================
def face_blink_system():
    known_encodings, known_names = load_known_faces()
    cam_index = find_camera()
    if cam_index == -1:
        print(" No camera found")
        return
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    EAR_THRESHOLD = 0.21
    CLOSED_FRAMES = 1
    face_data = {}
    next_face_id = 0
    print("\n Press Q to exit")
    last_person_count = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize for speed
        frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces
        face_locations = face_recognition.face_locations(rgb, model="hog")
        if len(face_locations) != last_person_count:
            print(f"Persons detected: {len(face_locations)}")
            last_person_count = len(face_locations)
        # Face encodings
        encodings = face_recognition.face_encodings(rgb, face_locations)
        # Landmarks
        landmarks_list = face_recognition.face_landmarks(rgb, face_locations)
        # ============================================
        # SHOW PERSON COUNT ON WINDOW
        # ============================================
        cv2.putText(
            frame,
            f"Persons Detected: {len(face_locations)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        # ============================================
        # PROCESS EACH FACE
        # ============================================
        for i, (top, right, bottom, left) in enumerate(face_locations):
            top = int(top)
            right = int(right)
            bottom = int(bottom)
            left = int(left)
            center = (
                (left + right) // 2,
                (top + bottom) // 2
            )
            # ============================================
            # SIMPLE FACE TRACKING
            # ============================================
            matched_id = None
            for fid in face_data:
                old_center = face_data[fid]["center"]
                dist = np.linalg.norm(
                    np.array(center) - np.array(old_center)
                )
                if dist < 50:
                    matched_id = fid
                    break
            # NEW FACE
            if matched_id is None:
                matched_id = next_face_id
                face_data[matched_id] = {
                    "center": center,
                    "blink_total": 0,
                    "state": "OPEN",
                    "frames_closed": 0,
                    "name": "Unknown"
                }
                next_face_id += 1
            data = face_data[matched_id]
            data["center"] = center
            # ============================================
            # FACE RECOGNITION
            # ============================================
            name = "Unknown"
            if i < len(encodings) and len(known_encodings) > 0:
                matches = face_recognition.compare_faces(
                    known_encodings,
                    encodings[i]
                )
                distances = face_recognition.face_distance(
                    known_encodings,
                    encodings[i]
                )
                best_match = np.argmin(distances)
                if matches[best_match]:
                    name = known_names[best_match]
            data["name"] = name
            # ============================================
            # DEFAULT STATUS
            # ============================================
            status = "No Eyes"
            color = (255, 255, 255)
            # ============================================
            # BLINK DETECTION
            # ============================================
            if i < len(landmarks_list):
                lm = landmarks_list[i]
                if "left_eye" in lm and "right_eye" in lm:
                    left_eye = np.array(lm["left_eye"])
                    right_eye = np.array(lm["right_eye"])
                    left_ear = EAR(left_eye)
                    right_ear = EAR(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    # ---------- STATE MACHINE ----------
                    if data["state"] == "OPEN":
                        if ear < EAR_THRESHOLD:
                            data["state"] = "CLOSED"
                            data["frames_closed"] = 1
                    elif data["state"] == "CLOSED":
                        if ear < EAR_THRESHOLD:
                            data["frames_closed"] += 1
                        else:
                            if data["frames_closed"] >= CLOSED_FRAMES:
                                data["blink_total"] += 1
                                print(
                                    f"ID {matched_id} ({name}) -> Blink #{data['blink_total']}"
                                )
                            data["state"] = "OPEN"
                            data["frames_closed"] = 0
                    status = data["state"]
                    if status == "OPEN":
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
            # ============================================
            # DRAW FACE BOX
            # ============================================
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )
            # Name + ID
            cv2.putText(
                frame,
                f"{name} (ID {matched_id})",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            # Eye Status
            cv2.putText(
                frame,
                status,
                (left, bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            # Blink Count
            cv2.putText(
                frame,
                f"Blinks: {data['blink_total']}",
                (left, bottom + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
        # ============================================
        # SHOW WINDOW
        # ============================================
        cv2.imshow("Face Recognition + Blink Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        # Q or ESC
        if key == ord('q') or key == 27:
            break
        # Window close button
        if cv2.getWindowProperty(
            "Face Recognition + Blink Detection",
            cv2.WND_PROP_VISIBLE
        ) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()

# ============================================
# MAIN MENU
# ============================================

def main():
    print("\n========== MENU ==========")
    print("1. Compare Two Images")
    print("2. Face + Blink Detection")
    print("==========================")
    choice = input("Enter choice: ")
    if choice == "1":
        img1 = input("Enter first image path: ")
        img2 = input("Enter second image path: ")
        compare_faces(img1, img2)
    elif choice == "2":
        face_blink_system()
    else:
        print(" Invalid choice")
# ============================================
# START PROGRAM
# ============================================

if __name__ == "__main__":
    main()
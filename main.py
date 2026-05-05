import face_recognition
import numpy as np
import cv2
from scipy.spatial.distance import cosine

# ============================================
# FACE COMPARISON FUNCTION
# ============================================

def compare_faces(image1_path, image2_path):
    try:
        print("\n Loading images...")

        img1 = face_recognition.load_image_file(image1_path)
        img2 = face_recognition.load_image_file(image2_path)

        loc1 = face_recognition.face_locations(img1)
        loc2 = face_recognition.face_locations(img2)

        if len(loc1) == 0:
            print(" No face found in:", image1_path)
            return
        if len(loc2) == 0:
            print(" No face found in:", image2_path)
            return

        enc1 = face_recognition.face_encodings(img1, loc1)[0]
        enc2 = face_recognition.face_encodings(img2, loc2)[0]

        print("\n128D Encoding (first 10 values):")
        print(enc1[:10])

        euclidean_dist = np.linalg.norm(enc1 - enc2)
        cos_sim = 1 - cosine(enc1, enc2)

        print("\n--- Results ---")
        print("Euclidean Distance:", round(euclidean_dist, 4))
        print("Cosine Similarity:", round(cos_sim, 4))

        if euclidean_dist < 0.45:
            print(" Same Person (High Confidence)")
        elif euclidean_dist < 0.65:
            print(" Maybe Same (Variation Possible)")
        else:
            print("Different Person")

    except Exception as e:
        print(" Error:", e)


# ============================================
# EAR FUNCTION
# ============================================

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


# ============================================
# BLINK DETECTION (FINAL STABLE VERSION)
# ============================================

def blink_detection():
    print("\n Opening Camera...")

    # Better camera handling for Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(" Camera not working")
        return

    cv2.namedWindow("Blink Detection", cv2.WINDOW_NORMAL)

    print(" Blink Detection Started")
    print(" Press Q / ESC OR close window  to exit")

    while True:
        ret, frame = cap.read()

        if not ret:
            print(" Frame not received")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            landmarks_list = face_recognition.face_landmarks(rgb)

            if len(landmarks_list) == 0:
                cv2.putText(frame, "No Face Detected", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

            for landmarks in landmarks_list:
                if 'left_eye' in landmarks and 'right_eye' in landmarks:

                    left_eye = np.array(landmarks['left_eye'])
                    right_eye = np.array(landmarks['right_eye'])

                    leftEAR = eye_aspect_ratio(left_eye)
                    rightEAR = eye_aspect_ratio(right_eye)

                    ear = (leftEAR + rightEAR) / 2.0

                    # Show EAR value
                    cv2.putText(frame, f"EAR: {round(ear, 2)}", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)

                    if ear < 0.25:
                        status = "Blink"
                    else:
                        status = "Eyes Open"

                    cv2.putText(frame, status, (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

        except Exception as e:
            print(" Error:", e)

        cv2.imshow("Blink Detection", frame)

        #  RELIABLE EXIT SYSTEM
        key = cv2.waitKey(1)

        if key != -1:
            print("Key pressed:", key)

        # Exit by keyboard
        if key == ord('q') or key == ord('Q') or key == 27:
            print("Exiting...")
            break

        # Exit by closing window
        if cv2.getWindowProperty("Blink Detection", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed")
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================
# MAIN PROGRAM
# ============================================

def main():
    print("\n===== FACE RECOGNITION SYSTEM =====")
    print("1. Compare Two Images")
    print("2. Blink Detection (EAR)")
    print("===================================")

    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        img1 = input("Enter first image path: ")
        img2 = input("Enter second image path: ")
        compare_faces(img1, img2)

    elif choice == "2":
        blink_detection()

    else:
        print(" Invalid choice")


# Run program
if __name__ == "__main__":
    print(" Program Started Successfully")
    main()
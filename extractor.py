import cv2
import mediapipe as mp
import numpy as np

class HolisticExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        # MediaPipe expects RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        
        # Extract coordinates
        # Pose: 33, Face: 468, Left Hand: 21, Right Hand: 21 -> Total 543
        def extract_landmarks(landmark_list, num_landmarks):
            if landmark_list:
                return np.array([[res.x, res.y, res.z] for res in landmark_list.landmark])
            else:
                return np.zeros((num_landmarks, 3))
                
        pose = extract_landmarks(results.pose_landmarks, 33)
        face = extract_landmarks(results.face_landmarks, 468)
        lh = extract_landmarks(results.left_hand_landmarks, 21)
        rh = extract_landmarks(results.right_hand_landmarks, 21)
        
        # Concatenate to match Kaggle Parquet alphabetical ordering:
        # 1. face (468)
        # 2. left_hand (21)
        # 3. pose (33)
        # 4. right_hand (21)
        landmarks = np.concatenate([face, lh, pose, rh])
        return landmarks, results

class WebcamStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.extractor = HolisticExtractor()
        
    def get_frame_data(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None
            
        landmarks, results = self.extractor.process_frame(frame)
        return landmarks, frame, results
        
    def release(self):
        self.cap.release()

if __name__ == "__main__":
    stream = WebcamStream()
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    
    print("Starting MediaPipe Extractor... Press ESC to exit.")
    
    while True:
        landmarks, frame, results = stream.get_frame_data()
        if frame is None:
            break
            
        # Draw landmark annotations
        image = frame.copy()
        
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        cv2.imshow('MediaPipe Holistic', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
            
    stream.release()
    cv2.destroyAllWindows()

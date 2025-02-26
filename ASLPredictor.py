import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


class ASLTranslator:


    def __init__(self, model_path):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils  # For visualization

        # Initialize OpenCV video capture
        self.cap = cv2.VideoCapture(0)

        # Creating dictionary
        self.letter = {0 : "A", 1 : "B", 2 : "C", 3 : "D", 4 : "DELETE", 5 : "E", 6 : "F", 7 : "G", 8 : "H", 9 : "I", 10 : "J", 11:"K", 12:"L", 13:"M", 14:"N", 15:"O", 16:"P", 17:"Q", 18:"R", 19:"S", 20:"SPACE",21:"T", 22:"U",23:"V", 24:"W", 25:"X", 26:"Y", 27:"Z"}
        """
        for i in range(26):
            self.letter[i] = str(chr(i + 65))
        """

    def process_frame(self, frame):
        """ Process a single frame to extract hand landmarks """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark positions
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z, lm.presence, lm.visibility])  # Flatten x, y, z values

                # Ensure correct input shape
                if len(landmarks) == 105:  # 21 landmarks * 3 coordinates
                    return np.array(landmarks).reshape(1, -1)

        return None  # No hand detected

    def predict(self):
        """ Continuously captures frames, processes them, and predicts ASL letters """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame to get landmarks
            landmarks = self.process_frame(frame)

            if landmarks is not None:
                # Make prediction
                prediction = self.model.predict(landmarks, verbose=0)
                predicted_label = np.argmax(prediction)  # Get class index

                # Display prediction on frame
                cv2.putText(frame, f'Prediction: {self.letter[predicted_label]}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Show webcam feed
            cv2.imshow('ASL Translator', frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# Usage
asl_translator = ASLTranslator('Dataset/asl_alphabet_train/MediapipeFFN.keras')
asl_translator.predict()
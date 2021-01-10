import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import numpy as np
import pyttsx3 

CLASSIFICATION = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
}


def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


model = load_model("model.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
)


class VideoCamera(object):
    def __init__(self):
        # capturing video
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # releasing camera
        self.video.release()

    def get_frame(self):
        while True:
            success, image = self.video.read()
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)
            # Preperation for bounding box

            image_height, image_width, _ = image.shape

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    
                    tup_x = (
                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                        * image_width,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x
                        * image_width,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x
                        * image_width,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
                        * image_width,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                        * image_width,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP
                        ].x
                        * image_width,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_PIP
                        ].x
                        * image_width,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_DIP
                        ].x
                        * image_width,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_TIP
                        ].x
                        * image_width,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_DIP
                        ].x
                        * image_width,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                        ].x
                        * image_width,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_PIP
                        ].x
                        * image_width,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                        ].x
                        * image_width,
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x
                        * image_width,
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x
                        * image_width,
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x
                        * image_width,
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x
                        * image_width,
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x
                        * image_width,
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x
                        * image_width,
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
                        * image_width,
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x
                        * image_width,
                    )

                    tup_y = (
                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                        * image_height,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y
                        * image_height,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
                        * image_height,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
                        * image_height,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
                        * image_height,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP
                        ].y
                        * image_height,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_PIP
                        ].y
                        * image_height,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_DIP
                        ].y
                        * image_height,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_TIP
                        ].y
                        * image_height,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_DIP
                        ].y
                        * image_height,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                        ].y
                        * image_height,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_PIP
                        ].y
                        * image_height,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                        ].y
                        * image_height,
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y
                        * image_height,
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
                        * image_height,
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
                        * image_height,
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
                        * image_height,
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
                        * image_height,
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
                        * image_height,
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
                        * image_height,
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y
                        * image_height,
                    )

                    y_ub = max(tup_y)  # Index finger
                    y_lb = min(tup_y)

                    x_ub = max(tup_x)
                    x_lb = min(tup_x)

                    w = abs(x_lb - x_ub)
                    h = abs(y_lb - y_ub)
                    x = min(tup_x) - 40
                    y = min(tup_y) - 40
                    image1 = image[int(x) : int(x + w + 80), int(y) : int(y + h + 80)]
                    img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (7172, 784))
                    imgs = np.array([img])
                    imgs = imgs.reshape(-1, 28, 28, 1)
                    model_output = model.predict(imgs)
                    
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    cv2.rectangle(
                        image,
                        (int(x), int(y)),
                        (int(x + w + 80), int(y + h + 80)),
                        (255, 0, 0),
                        2,
                    )
                    w = 0
                    if np.argmax(model_output) < 25:
                        w = np.argmax(model_output)
                    else:
                        w = 0
                    cv2.putText(
                        image,
                        CLASSIFICATION[w],
                        (50, 50),
                        font,
                        1,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    engine = pyttsx3.init()
                    engine.say(CLASSIFICATION[w])
                    engine.runAndWait()
                    success, jpeg = cv2.imencode(".jpeg", image)
                    return jpeg.tobytes()

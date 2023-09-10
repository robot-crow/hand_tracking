import cv2
import mediapipe as mp
import time

# what does the class have to do?
# take a frame, track the hands, send back an array

class HandTracker():
    def __init__(self):
        # self.mp_hands = mp.solutions.hands
        # self.hands = self.mp_hands.Hands()

        # get a hand processor
        self.hands = mp.solutions.hands.Hands()

    def track_hands(self, frame):
        # process the image for mand points detection
        results = self.hands.process(frame)

        return results




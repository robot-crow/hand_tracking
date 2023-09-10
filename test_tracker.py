import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os


class VisInputProcessor():
    def __init__(self):
        self.camera = 0
        self.video_capture = None

        # processing & visuals
        self.input_window = None

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def start_capture(self):
        self.video_capture = cv2.VideoCapture(self.camera)

        if not self.video_capture.isOpened():
            print("Error: Could not open video source.")
            return False

        #capture parameters
        desired_width = 640
        desired_height = 480
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

        self.input_window = cv2.namedWindow('DisplayWindow', cv2.WINDOW_NORMAL)

        return True


    def stop_capture(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
            self.input_window = None

    def process_frame(self, frame):
        # get landmarks to play with
        results = self.hands.process(frame)

        # the above returns false with no hands tracked
        if results.multi_hand_landmarks:
            # each hand detected becomes a set of landmarks. For each hand detected:
            for handLms in results.multi_hand_landmarks:
                # for each id-landmark pair in this hand
                # for id, lm in enumerate(handLms.landmark):
                #     px, py = int(lm.x * frame_w), int(lm.y * frame_h)

                self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def capture_frames(self):
        #for fps
        pTime = 0
        cTime = 0

        while True:
            ret, frame = self.video_capture.read()

            if not ret:
                break

            frame_h, frame_w, frame_c = frame.shape

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = self.process_frame(frame)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(frame, #where
                        str(int(fps)), #what
                        (5, 80), #pos
                        cv2.FONT_HERSHEY_PLAIN, #font
                        3, #size
                        (255, 0, 0), #colour
                        2) #thickness



            # Show the frame
            cv2.imshow('DisplayWindow', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow('DisplayWindow')

    def handle_input(self):
        # Start capturing video
        if self.start_capture():
            # Capture frames until 'q' key is pressed
            self.capture_frames()

        # Stop capturing video
        self.stop_capture()

def main():
    capture_manager = VisInputProcessor()
    capture_manager.handle_input()

if __name__ == '__main__':
    main()

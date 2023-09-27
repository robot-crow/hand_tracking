import os
import csv

import cv2
import mediapipe as mp
import numpy as np

import keyboard
import time

class GestureHarvester():
    def __init__(self, camera_index=0, cap_x=640, cap_y=480):
        self.camera_index = camera_index
        self.capturing = False
        self.cap_x = cap_x
        self.cap_y = cap_y

        self.mode = 0
        self.gesture_class = 0

        self.rec = False


        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils

        self.static_csv_path = os.path.join(os.getcwd(), 'models/static_gesture/data/gesture.csv')
        self.dynamic_csv_path = os.path.join(os.getcwd(), 'models/dynamic_gesture/data/gesture.csv')

    def key_check(self):
        # faster than waitkey at execution in this case

        if keyboard.is_pressed('q'):
            print('quitting')
            self.capturing = False
        elif keyboard.is_pressed('i'):
            self.mode = 0
        elif keyboard.is_pressed('o'):
            self.mode = 1
        elif keyboard.is_pressed('p'):
            self.mode = 2

        # this is verbose but runs with 0 framerate drops
        if keyboard.is_pressed('1'):
            self.gesture_class = 1
        elif keyboard.is_pressed('2'):
            self.gesture_class = 2
        elif keyboard.is_pressed('3'):
            self.gesture_class = 3
        elif keyboard.is_pressed('3'):
            self.gesture_class = 3
        elif keyboard.is_pressed('4'):
            self.gesture_class = 4
        elif keyboard.is_pressed('5'):
            self.gesture_class = 5
        elif keyboard.is_pressed('6'):
            self.gesture_class = 6
        elif keyboard.is_pressed('7'):
            self.gesture_class = 7
        elif keyboard.is_pressed('8'):
            self.gesture_class = 8
        elif keyboard.is_pressed('9'):
            self.gesture_class = 9
        elif keyboard.is_pressed('0'):
            self.gesture_class = 0

        if keyboard.is_pressed('space'):
            if self.mode != 0:
                self.rec = True

    def capture_process(self):
        # for fps
        pTime = 0
        cTime = 0
        counter = 0
        counter_max = 20
        fps = 0
        fps_list = []

        rec_counter = 0
        rec_max = 30

        # we only track one hand, as this is gesture recording for modelling
        hands = self.mpHands.Hands(static_image_mode=False,
                              max_num_hands=1,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5,)

        mpDraw = mp.solutions.drawing_utils

        # initialise video capture
        cap = cv2.VideoCapture(self.camera_index)

        cap_x = self.cap_x
        cap_y = self.cap_y
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_x)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_y)

        if not cap.isOpened():
            print('Error: Unable to open camera.')
            return
        else:
            print('Image capture started')

        # I hold captured gesture frames for both static and dynamic gestures
        gesture_buffer = []

        # do work
        while self.capturing:
            # single threaded process

            # rec will be 0 if space is not pressed.
            self.key_check()

            ret, frame = cap.read()

            if not ret:
                print('Error: Unable to read frame.')
                break

            # Mirror display
            frame = cv2.flip(frame, 1)

            hand_results = hands.process(frame)

            multi_hand_landmarks = hand_results.multi_hand_landmarks
            multi_handedness = hand_results.multi_handedness

            if self.rec == True:
                rec_counter = rec_counter + 1
                print(rec_counter)

                if rec_counter > rec_max:
                    self.rec = False
                    rec_counter = 0
                    self.log_csv(gesture_buffer)
                    # critical: wipe gesture buffer
                    gesture_buffer = []

            if multi_hand_landmarks:
                # each hand detected becomes a set of landmarks. For each hand detected (should only be one):
                for handLms, handed in zip(multi_hand_landmarks, multi_handedness):
                    # draw landmarks
                    mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

                    # get a bounding rectangle for lols
                    # brect is an array of x, y pos, and w, h
                    brect = self.calc_bounding_rect(cap_x, cap_y, handLms)
                    cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
                    hand_lr = handed.classification[0].label
                    cv2.putText(frame, hand_lr, (brect[0] + 5, brect[1] - 4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                    # process hand
                    proc_hand = self.proc_landmarks(handLms, handed)

                    if self.rec == True:

                        gesture_buffer.append(proc_hand)

            # show mode
            if self.mode == 0:
                mode = 'display only'
            elif self.mode == 1:
                mode = 'capture static gesture'
            elif self.mode == 2:
                mode = 'capture mobile gesture'

            cv2.putText(frame,  # where
                        mode,  # what
                        (5, 35),  # pos
                        cv2.FONT_HERSHEY_PLAIN,  # font
                        1,  # size
                        (0, 0, 255),  # colour
                        2)  # thickness

            if self.mode > 0:
                gesture = 'gesture_class: ' + str(self.gesture_class)
                cv2.putText(frame,  # where
                            gesture,  # what
                            (5, 50),  # pos
                            cv2.FONT_HERSHEY_PLAIN,  # font
                            1,  # size
                            (0, 0, 255),  # colour
                            2)  # thickness

            # current time, time delta since last time, processing time
            cTime = time.time()
            dTime = (cTime - pTime)
            pTime = cTime
            counter += 1

            # in very rare cases processing occurs so fast that dTime is closer to 0 than precision maybe
            if dTime > 0:
                fps_run = 1 / dTime
                fps_list.append(fps_run)

            # every 10 cycles update fps (make readable)
            if counter % counter_max == 0:
                fps = int(np.mean(fps_list))
                counter = 0
                fps_list = []

                # critical: wipe static gesture buffer
                static_det_buffer = []

            proc_speed = 'proc_fps: ' + str(fps)

            # show fps
            cv2.putText(frame,  # where
                        proc_speed,  # what
                        (5, 20),  # pos
                        cv2.FONT_HERSHEY_PLAIN,  # font
                        1,  # size
                        (0, 0, 255),  # colour
                        2)  # thickness

            cv2.imshow('output_img', frame)
            cv2.waitKey(1)


        # hand back resources
        cap.release()

    def calc_bounding_rect(self, image_width, image_height, landmarks):
        # Extract landmark coordinates into a list of tuples
        landmark_points = [(int(landmark.x * image_width), int(landmark.y * image_height)) for landmark in
                           landmarks.landmark]

        # Convert the list of tuples to a NumPy array
        landmark_array = np.array(landmark_points, dtype=np.int32)

        # Calculate the bounding rectangle
        x, y, w, h = cv2.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def proc_landmarks(self, handLms, handed):
        # I require that I actually be given a valid landmark object for a single hand
        # I discard the z values as they are not meaningful without actual depth measurements
        # I wish to normalise x and y such that any hand exists in a square space
        handed = handed.classification[0].index

        lms_list = [[lm.x, lm.y] for lm in handLms.landmark].copy()
        base_x, base_y = lms_list[0][0], lms_list[0][1]

        lms_ran = range(0, len(lms_list))

        lms_list = [[lms_list[i][0] - base_x, lms_list[i][1] - base_y] for i in lms_ran]

        max_x = max(map(abs, [lms_list[i][0] for i in lms_ran]))
        max_y = max(map(abs, [lms_list[i][1] for i in lms_ran]))

        lms_tform = [[lm[0] / max_x, lm[1] / max_y] for lm in lms_list]

        # insert the handed value at pos 0. Right == 1
        hand_meta = [self.gesture_class, handed]
        lms_tform = np.insert(lms_tform, 0, hand_meta)

        lms_tform = np.array(lms_tform).flatten().tolist()

        return lms_tform

    def log_csv(self, gesture_buffer):
        # Open the CSV file in write mode
        if self.mode == 1:
            with open(self.static_csv_path, mode='a', newline='') as file:
                # Create a csv.writer object
                csv_writer = csv.writer(file)

                for proc_hand in gesture_buffer:
                    # Write the list as a single row
                    csv_writer.writerow(proc_hand)

        elif self.mode == 2:
            with open(self.dynamic_csv_path, mode='a', newline='') as file:
                # Create a csv.writer object
                csv_writer = csv.writer(file)

                for proc_hand in gesture_buffer:
                    # Write the list as a single row
                    csv_writer.writerow(proc_hand)


        return

    def run(self):
        print('GestureHarvester starting up...')
        self.capturing = True
        self.capture_process()

if __name__ == '__main__':
    gh = GestureHarvester()
    gh.run()
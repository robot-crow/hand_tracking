import cv2
import mediapipe as mp
import numpy as np
import csv

import multiprocess as mupr

import time
import keyboard

from models import StaticGestureClassifier
from models import DynamicGestureClassifier
from collections import deque
from collections import Counter

class VisInputProcessor():
    def __init__(self, camera_index=0, cap_x=640, cap_y=480):
        # a process captures and puts in capture_queue.
        # Another process does detection and puts to process_queue
        self.camera_index = camera_index
        self.cap_x = cap_x
        self.cap_y = cap_y


        self.capture_queue = mupr.Queue(maxsize=1)
        self.process_queue = mupr.Queue(maxsize=1)

        self.quit_capture = mupr.Event()
        self.quit_process = mupr.Event()
        self.quit_display = mupr.Event()

        self.mpHands = mp.solutions.hands

        self.static_class_labels_path = 'models/static_gesture/static_class_labels.csv'

        with open(self.static_class_labels_path, encoding='utf-8-sig') as f:
            static_class_labels = csv.reader(f)
            self.static_class_labels = [row[0] for row in static_class_labels]

        self.dynamic_class_labels_path = 'models/dynamic_gesture/dynamic_class_labels.csv'

        with open(self.dynamic_class_labels_path, encoding='utf-8-sig') as f:
            dynamic_class_labels = csv.reader(f)
            self.dynamic_class_labels = [row[0] for row in dynamic_class_labels]

    def start_capture(self):
        self.capture_thread = mupr.Process(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def stop_capture(self):
        self.quit_capture.set()
        self.capture_thread.join()

    def start_process(self):
        self.process_thread = mupr.Process(target=self._process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()

    def stop_process(self):
        self.quit_process.set()
        self.process_thread.join()

    def start_display(self):
        self.display_thread = mupr.Process(target=self._display_output)
        self.display_thread.daemon = True
        self.display_thread.start()

    def stop_display(self):
        self.quit_display.set()
        self.display_thread.join()


    def _capture_frames(self):

        cap = cv2.VideoCapture(self.camera_index)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cap_x)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_y)

        if not cap.isOpened():
            print('Error: Unable to open camera.')
            return

        while not self.quit_capture.is_set():
            ret, frame = cap.read()

            if ret:
                if not self.capture_queue.full():
                    self.capture_queue.put(frame, block=False)
            else:
                print('Error: Unable to read frame.')
                break

        cap.release()

    def _process_frames(self):
        #for fps
        pTime = 0
        cTime = 0
        counter = 0
        fps = 0
        fps_list = []

        hands = self.mpHands.Hands(static_image_mode=False,
                              max_num_hands=2,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5,)

        while not self.quit_process.is_set():
            if not self.capture_queue.empty() and not self.process_queue.full():
                frame = self.capture_queue.get(block=False)
                frame = cv2.flip(frame, 1)  # Mirror display
                # frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)

                # get landmarks to play with
                hand_results = hands.process(frame)

                # processing end


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
                if counter % 10 == 0:
                    fps = int(np.mean(fps_list))
                    counter = 0
                    fps_list = []

                # tuple in order:
                # [0]=image, [1]=fps, [2]=coords hands, [3]=handedness (0 is left hand)

                frame_tuple = (frame, fps, hand_results.multi_hand_landmarks, hand_results.multi_handedness)

                # push one to the queue

                self.process_queue.put(frame_tuple, block=False)

    def _display_output(self):
        # I take a dict of images and display them at specific places
        # cv2.namedWindow('Captured Image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('output_img', cv2.WINDOW_AUTOSIZE)
        mpDraw = mp.solutions.drawing_utils

        static_gesture_classifier = StaticGestureClassifier()
        dynamic_gesture_classifier = DynamicGestureClassifier()

        # dynamic gesture buffer - gesture size is ALWAYS 30 frames.
        # as dynamic gestures will need several scan passes, add 10 frames and
        # use a moving window, take most common classfication for gesture
        empty_buf = [None] * 40
        left_buf = deque(empty_buf.copy(), len(empty_buf))
        right_buf = deque(empty_buf.copy(), len(empty_buf))

        while not self.quit_display.is_set():
            if not self.process_queue.empty():
                frame_tuple = self.process_queue.get(block=True, timeout=1)

                frame = frame_tuple[0]
                fps = frame_tuple[1]
                hand_landmarks = frame_tuple[2]
                handedness = frame_tuple[3]

                left_val = None
                right_val = None

                # the above returns false with no hands tracked
                if hand_landmarks:
                    # each hand detected becomes a set of landmarks. For each hand detected:
                    for handLms, handed in zip(hand_landmarks, handedness):
                        mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

                        # bounding rectangle brect is an array of x, y pos, and w, h
                        brect = self.calc_bounding_rect(self.cap_x, self.cap_y, handLms)
                        cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
                        handlr = handed.classification[0].label

                        # get handedness index. 1 is right, 0 is left.
                        handed = handed.classification[0].index

                        # process static hand
                        proc_hand = self.proc_landmarks(handLms, handed)

                        # handle dynamic buffer - these could be in ONE BUFFEr but I HAVENT TIME
                        dynamic_gesture = 0
                        match handed:
                            case 0:
                                # first, put this hand into left_val
                                left_val = (handLms, handed)

                                # now attempt dynamic gesture detection
                                if not any(item is None for item in left_buf):
                                    print('left queue valid')
                                    dynamic_gesture = self.scan_buffer(left_buf, dynamic_gesture_classifier)

                            case 1:
                                # first, put this hand into right_val
                                right_val = (handLms, handed)

                                # now attempt dynamic gesture detection
                                if not any(item is None for item in right_buf):
                                    print('right queue valid')
                                    dynamic_gesture = self.scan_buffer(right_buf, dynamic_gesture_classifier)

                        # classify static hand gesture
                        static_gesture = static_gesture_classifier(proc_hand)
                        
                        hand_text_static = str(handlr) + ' : ' + str(self.static_class_labels[static_gesture])
                        hand_text_dynamic = 'Gesture : ' + str(self.dynamic_class_labels[dynamic_gesture])

                        cv2.putText(frame, hand_text_static, (brect[0] + 5, brect[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                        cv2.putText(frame, hand_text_dynamic, (brect[0] + 5, brect[1] - 54),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    left_val = None
                    right_val = None

                # dynamic buffer
                # pop one from the queue
                # push the left/right detection states in
                left_buf.popleft()
                left_buf.append(left_val)

                right_buf.popleft()
                right_buf.append(right_val)

                # if not any(item is None for item in left_buf):
                #     print('left queue valid')
                #     left_gesture = self.scan_buffer(left_buf, dynamic_gesture_classifier)

                # if not any(item is None for item in right_buf):
                #     print('right queue valid')
                #     right_gesture = self.scan_buffer(right_buf, dynamic_gesture_classifier)



                proc_speed = 'proc_fps: ' + str(fps)

                cv2.putText(frame,  # where
                           proc_speed,  # what
                           (5, 80),  # pos
                           cv2.FONT_HERSHEY_PLAIN,  # font
                           1,  # size
                           (0, 0, 255),  # colour
                           2)  # thickness

                cv2.imshow('output_img', frame)
                cv2.waitKey(1)

        cv2.destroyWindow('output_img')


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

        lms_list = [[lm.x, lm.y] for lm in handLms.landmark].copy()

        #the first points here are X and Y for the base of the hand
        base_x, base_y = lms_list[0][0], lms_list[0][1]

        lms_ran = range(0, len(lms_list))

        # ith entries are landmarks, [0] is the X, [1] is the Y
        lms_list = [[lms_list[i][0] - base_x, lms_list[i][1] - base_y] for i in lms_ran]

        # give me the maximum value of a mapped absolute for each X and Y BUT DO NOT MODIFY THEM
        max_x = max(map(abs, [lms_list[i][0] for i in lms_ran]))
        max_y = max(map(abs, [lms_list[i][1] for i in lms_ran]))

        # dividing negative by positive numbers is fine....
        lms_tform = [[lm[0] / max_x, lm[1] / max_y] for lm in lms_list]

        # insert the handed value at pos 0. Right == 1. Similar to func in app_harvest_dynamic
        hand_meta = [handed]
        lms_tform = np.insert(lms_tform, 0, hand_meta)

        lms_tform = np.array(lms_tform).flatten().tolist()

        return lms_tform

    def get_buffer_minmax(self, gesture_buffer):
        min_x = 0
        min_y = 0
        max_x = 0
        max_y = 0

        for i, buf_tuple in enumerate(gesture_buffer):
            handLms = buf_tuple[0]
            handed = buf_tuple[1]
            lms_list = [[lm.x, lm.y] for lm in handLms.landmark].copy()
            lms_ran = range(0, len(lms_list))

            # find the top left global coords of the gesture
            x_ = [lms_list[i][0] for i in lms_ran]
            y_ = [lms_list[i][1] for i in lms_ran]

            min_x_ = min(map(abs, x_))
            min_y_ = min(map(abs, y_))
            max_x_ = max(map(abs, x_))
            max_y_ = max(map(abs, y_))

            if i == 0:
                min_x = min_x_
                min_y = min_y_
                max_x = max_x_
                max_y = max_y_
            else:
                if min_x_ < min_x:
                    min_x = min_x_
                if min_y_ < min_y:
                    min_y = min_y_
                if max_x_ > max_x:
                    max_x = max_x_
                if max_y_ > max_y:
                    max_y = max_y_

        return min_x, min_y, max_x, max_y

    def proc_gesture_buffer(self, gesture_buffer):
        gesture_buffer_tform = []
        # first off we need the max xy and min xy across all frames
        min_x, min_y, max_x, max_y = self.get_buffer_minmax(gesture_buffer.copy())

        # subtract the min from the max to avoid stretching the gesture
        max_x = max_x - min_x
        max_y = max_y - min_y

        for i, buf_tuple in enumerate(gesture_buffer):
            handLms = buf_tuple[0]
            handed = buf_tuple[1]
            lms_list = [[lm.x, lm.y] for lm in handLms.landmark]
            lms_ran = range(0, len(lms_list))

            lms_list = [[lms_list[i][0] - min_x, lms_list[i][1] - min_y] for i in lms_ran]
            lms_tform = [[lm[0] / max_x, lm[1] / max_y] for lm in lms_list]

            # insert the handed value at pos 0. Right == 1
            hand_meta = [handed]
            lms_tform = np.insert(lms_tform, 0, hand_meta)
            lms_tform = np.array(lms_tform).flatten().tolist()

            gesture_buffer_tform.append(lms_tform)

        return gesture_buffer_tform

    # my job is to be called periodicially on a buffer and establish the most common dynamic gesture found
    def scan_buffer(self, gesture_buffer, dynamic_gesture_classifier):
        gesture_list = []
        gesture_buffer_tform = self.proc_gesture_buffer(gesture_buffer)
        # print(gesture_buffer_tform)

        # for frame in gesture_buffer_tform:
        window_size = 30
        for i in range(0, len(gesture_buffer_tform) - window_size):

            gesture_window = gesture_buffer_tform[i: i + window_size]
            gesture = dynamic_gesture_classifier(gesture_window)
            gesture_list.append(gesture)

        gesture = Counter(gesture_list).most_common()[0][0]
        return gesture

    def close(self):
        self.stop_capture()
        self.stop_process()
        self.stop_display()

    def run(self):
        self.start_capture()
        self.start_process()
        self.start_display()


        while True:
            if keyboard.is_pressed('q'):
                print('quitting')

                self.close()

                print('success')
                break




if __name__ == '__main__':
    capture_manager = VisInputProcessor()
    capture_manager.run()
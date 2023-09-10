import os
import time
import cv2
import multiprocess
import keyboard

class ImageCaptureDisplay:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.capture_queue = multiprocess.Queue(maxsize=1)
        self.process_queue = multiprocess.Queue(maxsize=1)

        # flags to track state
        self.capture_active = False
        self.process_active = False
        self.display_active = False

    # simple
    def start_capture(self):
        self.capture_active = True
        self.capture_thread = multiprocess.Process(target=self._capture_images)
        # self.capture_thread.daemon = True
        self.capture_thread.start()

    def stop_capture(self):
        self.capture_active = False
        self.capture_thread.join()
        print('capture stopped')

    def start_processing(self):
        self.process_active = True
        self.process_thread = multiprocess.Process(target=self._process_images)
        # self.process_thread.daemon = True
        self.process_thread.start()

    def stop_processing(self):
        print('proc stopped')
        self.process_active = False
        self.process_thread.join()

    def start_display(self):
        self.display_active = True
        self.display_thread = multiprocess.Process(target=self._display_images)
        # self.display_thread.daemon = True
        self.display_thread.start()

    def stop_display(self):
        print('display stopped')
        self.display_active = False
        self.display_thread.join()

    def _capture_images(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print('Error: Unable to open camera.')
            return

        while self.capture_active:
            ret, frame = cap.read()

            if ret:
                if not self.capture_queue.full():
                    self.capture_queue.put(frame, block=False)
            else:
                print('Error: Unable to read frame.')
                break
        print('foo')
        cap.release()

    def _process_images(self):
        while self.process_active:
            if not self.capture_queue.empty():
                frame = self.capture_queue.get(block=True)
                if not self.process_queue.full():
                    self.process_queue.put(frame, block=False)



    def _display_images(self):
        cv2.namedWindow('Captured Image', cv2.WINDOW_NORMAL)
        while self.display_active:
           if not self.process_queue.empty():
               frame = self.process_queue.get(block=True, timeout=1)
               cv2.imshow('Captured Image', frame)
               cv2.waitKey(1)

        cv2.destroyWindow('Captured Image')

    def close(self):
        print('stopping')
        self.stop_capture()
        self.stop_processing()
        self.stop_display()
        cv2.destroyAllWindows()


def main():
    cam_display = ImageCaptureDisplay()
    cam_display.start_capture()
    cam_display.start_processing()
    cam_display.start_display()

    while True:

        if keyboard.is_pressed('q'):
            print('troo')
            cam_display.close()
            break


if __name__ == '__main__':
    main()


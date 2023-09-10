import cv2
import numpy as np
import multiprocess as mupr
import keyboard


class VisInputProcessor():
    def __init__(self,camera_index=0):
        self.capture = False
        self.process= False
        self.display = False

        self.camera_index = camera_index
        self.capture_queue = mupr.Queue(maxsize=1)
        self.process_queue = mupr.Queue(maxsize=1)

        self.quit_capture = mupr.Event()
        self.quit_process = mupr.Event()
        self.quit_display = mupr.Event()

    def start_capture(self):
        self.capture = True
        self.capture_thread = mupr.Process(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def stop_capture(self):
        self.capture = False
        self.quit_capture.set()
        self.capture_thread.join()

    def start_process(self):
        self.process = True
        self.process_thread = mupr.Process(target=self._process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()

    def stop_process(self):
        self.process = False
        self.quit_process.set()
        self.process_thread.join()

    def start_display(self):
        self.display = True
        self.display_thread = mupr.Process(target=self._display_output)
        self.display_thread.daemon = True
        self.display_thread.start()

    def stop_display(self):
        self.display = False
        self.quit_display.set()
        self.display_thread.join()


    def _capture_frames(self):

        cap = cv2.VideoCapture(self.camera_index)
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

    # def _process_frames(self):
    #     while not self.quit_process.is_set():
    #         if not self.capture_queue.empty():
    #             frame = self.capture_queue.get(block=True)
    #             if not self.process_queue.full():
    #                 self.process_queue.put(frame, block=False)

    def _process_frames(self):
        while not self.quit_process.is_set():
            if not self.capture_queue.empty() and not self.process_queue.full():
                frame_in = self.capture_queue.get(block=True)
                # perform actual processing
                frame_out = frame_in.copy()

                # Get the dimensions of the images
                height, width, _ = frame_in.shape

                # Create a canvas to display the images side by side
                canvas_width = 2 * width  # Adjust this according to your needs
                canvas_height = height
                canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

                # Place the images onto the canvas at specific positions
                canvas[:height, :width] = frame_in
                canvas[:height, width:] = frame_out

                self.process_queue.put(canvas, block=False)

    def _display_output(self):
        # cv2.namedWindow('Captured Image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('output_img', cv2.WINDOW_AUTOSIZE)

        while not self.quit_display.is_set():
           if not self.process_queue.empty():
               frame = self.process_queue.get(block=True, timeout=1)
               cv2.imshow('output_img', frame)
               cv2.waitKey(1)

        cv2.destroyWindow('output_img')

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
                self.capture = False
                self.process = False
                self.display = False

                self.close()


                # if self.capture_thread.is_alive():
                #     print('capalive')
                # if self.process_thread.is_alive():
                #     print('procalive')
                # if self.display_thread.is_alive():
                #     print('dispalive')

                print('success')
                break




if __name__ == '__main__':
    capture_manager = VisInputProcessor()
    capture_manager.run()


# import cv2
# import multiprocessing
# import numpy as np
#
# class ImageProcessor:
#     def __init__(self):
#         self.capture_process = None
#         self.processing_process = None
#         self.display_process = None
#         self.quit_event = multiprocessing.Event()
#         self.frame_queue = multiprocessing.Queue()
#         self.processed_frame_queue = multiprocessing.Queue()
#
#     def start(self):
#         self.capture_process = multiprocessing.Process(target=self.capture_images)
#         self.processing_process = multiprocessing.Process(target=self.process_images)
#         self.display_process = multiprocessing.Process(target=self.display_images)
#
#         self.capture_process.start()
#         self.processing_process.start()
#         self.display_process.start()
#
#         self.capture_process.join()
#         self.processing_process.join()
#         self.display_process.join()
#
#     def capture_images(self):
#         cap = cv2.VideoCapture(0)  # Open the default camera (usually laptop webcam)
#
#         while not self.quit_event.is_set():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             # Put the captured frame into a queue for processing
#             self.frame_queue.put(frame)
#
#         cap.release()
#
#     def process_images(self):
#         while not self.quit_event.is_set():
#             if not self.frame_queue.empty():
#                 frame = self.frame_queue.get()
#
#                 # Perform image processing here (e.g., apply filters)
#                 processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#                 # Put the processed frame into a queue for display
#                 self.processed_frame_queue.put(processed_frame)
#
#     def display_images(self):
#         while not self.quit_event.is_set():
#             if not self.processed_frame_queue.empty():
#                 processed_frame = self.processed_frame_queue.get()
#
#                 # Display the processed frame
#                 cv2.imshow("Processed Frame", processed_frame)
#
#                 # Check for the 'q' key press to quit
#                 key = cv2.waitKey(1)
#                 if key == ord('q'):
#                     self.quit_event.set()
#                     break
#
#         cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     image_processor = ImageProcessor()
#     image_processor.start()
import numpy as np
import tensorflow as tf


class DynamicGestureClassifier(object):
    def __init__(self, model_path='models/dynamic_gesture/dynamic_classifier.tflite', num_threads=1):

        # load the tflite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    # This class does one thing: load and query a model to classify a gesture. __call__ is appropriate
    def __call__(self,landmark_list
                 ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_details_tensor_index, np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))

        return result_index
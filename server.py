import os
import io
import socket
import struct
import cv2
import numpy as np
import time
# import matplotlib.pyplot as pl
from tensorflow.lite.python.interpreter import Interpreter
# from tflite_runtime.interpreter import Interpreter

# Binary data format stands for how to read the incoming byte array
# '<' depends on host system and stands for little-endian (compatible for Intelx86 and AMD64)
# 'L' stands for long (32-bit unsigned integer) in the C language data type
# For more info : https://docs.python.org/3/library/struct.html
BINARY_DATA_FORMAT = '<L'

# The IP address of the server -> open command line and type 'ipconfig'. Get your ipv4
IP_ADDRESS = '192.168.2.116'

# The port to be listened on the server
PORT = 8000

# Get paths
CWD_PATH = os.getcwd() # working directory
MODEL_PATH = CWD_PATH + os.path.sep + 'model' + os.path.sep + 'detect.tflite' # directory of object detection model
LABELMAP_PATH = CWD_PATH + os.path.sep + 'model' + os.path.sep + 'labelmap.txt' # directory of object labes

# Confidence threshold for detected objects
CONFIDENCE_THRESHOLD = 0.5

# Load the label map // Neglect lines with '???'
labels = []
with open(LABELMAP_PATH, 'r') as f:
    while True:

        line = f.readline()
        if not line:
            break
        line = line.strip()

        if line != '???':
            labels.append(line)

# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Create a server side socket and start to listen
server_socket = socket.socket()
server_socket.bind((IP_ADDRESS, PORT))
server_socket.listen(0)

# Accept a client and create connection
connection = server_socket.accept()[0].makefile('rb') # rb stands for 'read'
try:
    while True:

        # Get the buffer length of to be received data & Quit if it is zero
        buffer_length = struct.unpack(BINARY_DATA_FORMAT, connection.read(struct.calcsize(BINARY_DATA_FORMAT)))[0]
        if not buffer_length:
            break

        # Write the image coming from client to stream
        image_stream = io.BytesIO()
        image_stream.write(connection.read(buffer_length))

        # Go to stream beginning
        image_stream.seek(0)

        # Convert the image stream to byte array
        data = np.frombuffer(image_stream.getvalue(), dtype=np.uint8)

        # Create an image out of byte array
        frame_orig = cv2.imdecode(data, 1)
        imageHeight = frame_orig.shape[0]
        imageWidth = frame_orig.shape[1]

        frame = frame_orig.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        
        # Convert 3 dimensional array into 4 for interpreter
        input_data = np.expand_dims(frame_resized, axis=0)

        # Set tensor for tensorflow and run object detection on frame (data)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if (classes[i] == 0) and ((scores[i] > CONFIDENCE_THRESHOLD) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imageHeight)))
                xmin = int(max(1,(boxes[i][1] * imageWidth)))
                ymax = int(min(imageHeight,(boxes[i][2] * imageHeight)))
                xmax = int(min(imageWidth,(boxes[i][3] * imageWidth)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        
        # img = cv2.imread('image.jpg')
        cv2.imshow('IMAGE', frame)

        # Press 'Q' on keyboard to quit
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    connection.close()
    server_socket.close()
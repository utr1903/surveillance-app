import io
import socket
import struct
import time
import picamera

# Binary data format stands for how to read the incoming byte array
# '<' depends on host system and stands for little-endian (compatible for Intelx86 and AMD64)
# 'L' stands for long (32-bit unsigned integer) in the C language data type
# For more info : https://docs.python.org/3/library/struct.html
BINARY_DATA_FORMAT = '<L'

# The IP address of the server -> open command line and type 'ipconfig'. Get your ipv4
IP_ADDRESS = '192.168.2.116'

# The port to be listened on the server
PORT = 8000

# Create a socket and connect to server
client_socket = socket.socket()
client_socket.connect((IP_ADDRESS, PORT))

# Create a connection to write the stream
connection = client_socket.makefile('wb') # wb stands for 'write'
try:
    # Initialize Pi camera
    camera = picamera.PiCamera()
    camera.vflip = True
    camera.resolution = (500, 480)

    # Start a preview and let the camera warm up for 2 seconds
    camera.start_preview()
    time.sleep(2)

    # Create a stream to write the camera input
    stream = io.BytesIO()

    # Capture image continuously (record a video into the stream)
    for _ in camera.capture_continuous(stream, 'jpeg'):

        # Send the length of to be sent image
        connection.write(struct.pack(BINARY_DATA_FORMAT, stream.tell()))
        connection.flush()

        # Go to the stream beginning and send the whole image content
        stream.seek(0)
        connection.write(stream.read())

        # Reset the stream for the next capture
        stream.seek(0)
        stream.truncate()

    # Write a length of zero into the stream to terminate
    connection.write(struct.pack(BINARY_DATA_FORMAT, 0))

finally:
    connection.close()
    client_socket.close()

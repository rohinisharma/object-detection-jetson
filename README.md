# Object Detection For Video Stream on An NVIDIA Jetson board
TensorflowObjectDetectionpictures.ipynb: An ipython notebook that uses OpenCV and a frozen tensorflow model to perform 
object detection on a single image

canny.ipynb: An ipython notebook that uses OpenCV transform each frame in a video stream to a frame of just its edges, using canny edge detection in OpenCV

obj-det-fps.py: A script that uses OpenCV and a frozen tensorflow model to detect objects that appear in the frame of a video stream, label these objects, and draw a bounding box around them. 


Dependencies for obj-det-fps.py:
- Tensorflow
- OpenCV
- imutils

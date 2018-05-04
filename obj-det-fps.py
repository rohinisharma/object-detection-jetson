import numpy as np
import os 
import cv2 
import time 
import argparse
import multiprocessing
import tensorflow as tf
from multiprocessing import Queue, Pool
from imutils.video import WebcamVideoStream
from imutils.video import FPS
#from matplotlib import pyplot as plt
import sys
#import six.moves.urllib as urllib
import tarfile
from PIL import Image
import cv2
#Dependencies: OpenCV, Tensorflow, imutils
sys.path.append("/home/nvidia/Documents/object_detection/")
sys.path.append("..")
 
from utils import label_map_util
from utils import visualization_utils as vis_util
detection_graph =tf.Graph()
with tf.device('device:GPU:0'):
	with detection_graph.as_default():
	    od_graph_def = tf.GraphDef()
	    with tf.gfile.GFile("/home/nvidia/Documents/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb", 
		                'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def,name='')
	NUM_CLASSES=90
	label_map = label_map_util.load_labelmap("/home/nvidia/Documents/object_detection/data/mscoco_label_map.pbtxt")
	categories = label_map_util.convert_label_map_to_categories(label_map, 
		                                                    max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)
	if __name__ == '__main__':
	    parser = argparse.ArgumentParser()
	    parser.add_argument('-src', '--source', dest='video_source', type=int,
		                default=0, help='Device index of the camera.')
	    parser.add_argument('-wd', '--width', dest='width', type=int,
		                default=480, help='Width of the frames in the video stream.')
	    parser.add_argument('-ht', '--height', dest='height', type=int,
		                default=360, help='Height of the frames in the video stream.')
	    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
		                default=2, help='Number of workers.')
	    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
		                default=5, help='Size of the queue.')
	    args = parser.parse_args()

	  #  input_q = Queue(maxsize=args.queue_size)
    	   # output_q = Queue(maxsize=args.queue_size)
    	   # pool = Pool(args.num_workers, worker, (input_q, output_q))	

	    
	    video_capture = WebcamVideoStream(src=1).start()
	    fps = FPS().start()
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with detection_graph.as_default():
	    with tf.Session(config = config,graph=detection_graph) as sess:
		while True:
		    image_np = video_capture.read()
		#input and output tensors for detection_graph
		    image_tensor=detection_graph.get_tensor_by_name("image_tensor:0")
		    detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
		    detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
		    detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
		    num_detections = detection_graph.get_tensor_by_name("num_detections:0")
		
		    
		    image_np_expanded = np.expand_dims(image_np,axis=0)
		    (boxes,scores,classes,num)= sess.run(
		            [detection_boxes, detection_scores, detection_classes, num_detections],
		    feed_dict={image_tensor: image_np_expanded})
		    vis_util.visualize_boxes_and_labels_on_image_array(
		          image_np,
		          np.squeeze(boxes),
		          np.squeeze(classes).astype(np.int32),
		          np.squeeze(scores),
		          category_index,
		        use_normalized_coordinates=True, line_thickness=8)
		    cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
		    if cv2.waitKey(25) & 0xFF == ord('q'):
		       cv2.destroyAllWindows()
		       break
		    
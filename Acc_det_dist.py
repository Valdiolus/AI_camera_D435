#MIT license by Valdis Gerasymiak
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import math

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}


RPI = 0
RSD = 0
TPU = 0
VIDEO = 0
TRACKER = 0

import pyrealsense2 as rs

if(TPU != 0):
	from edgetpu.detection.engine import DetectionEngine
	from PIL import Image

if(RPI == 1):
	from picamera.array import PiRGBArray
	from picamera import PiCamera

accuracy = 0.3

if(TRACKER == 1):
	from sort_master.sort import *

if(TRACKER == 2):
	#initialize OpenCV's special multi-object tracker
	aa=1

def USB_ONBOARD_CAMERA_init(ch):
	# initialize the video stream, allow the cammera sensor to warmup,
	# and initialize the FPS counter
	print("[INFO] starting video stream...")
	vs = VideoStream(ch).start()
	time.sleep(0.5)
	return vs

def USB_ONBOARD_CAMERA_get(stream_int):
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame_int = stream_int.read()
	#frame_int = imutils.resize(frame_int, width=400)
	return frame_int

def RPI_CAMERA_init(res, fr):
	camera = PiCamera()
	camera.resolution = res
	camera.framerate = fr
	rawCapture = PiRGBArray(camera, size=res)
	time.sleep(0.1)
	return camera

def RPI_CAMERA_get(stream_int, output_int):
	stream_int.capture(output_int, 'rgb')
	return output_int

def VIDEO_init(file):
	vs = cv2.VideoCapture(file)
	if (vs.isOpened() == False):
		print("Error opening video stream or file")
	time.sleep(0.5)
	return vs

def VIDEO_get(stream_int):
	ret, frame_int = stream_int.read()
	#frame_int = imutils.resize(frame_int, width=400)
	return frame_int

def RS_D435_init(pipeline_int):
	# Configure depth and color streams
	config = rs.config()
	config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
	config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

	# Start streaming
	pipeline_int.start(config)

def RS_D435_get(pipeline_int):
	# Wait for a coherent pair of frames: depth and color
	frames = pipeline_int.wait_for_frames()
	depth_frame = frames.get_depth_frame()
	color_frame = frames.get_color_frame()
	if not depth_frame or not color_frame:
		return (0,0,0), (0,0,0), (0,0,0)

	# Convert images to numpy arrays
	depth_image_int = np.asanyarray(depth_frame.get_data())
	color_image_int = np.asanyarray(color_frame.get_data())

	# dnn
	#im = cv2.resize(color_image_int, (300, 300))
	#im = im - 127.5
	#im = im * 0.007843
	#im.astype(np.float32)
	return color_image_int, depth_image_int, depth_frame #if need im or color image???

# Function to read labels from text files.
def ReadLabelFile(file_path):
	with open(file_path, 'r') as f:
		lines = f.readlines()
	ret = {}
	for line in lines:
		pair = line.strip().split(maxsplit=1)
		ret[int(pair[0])] = pair[1].strip()
	return ret


def LOAD_CPU_CAFFE_MnetSSD():
	# construct the argument parse and parse the arguments
	# initialize the list of class labels MobileNet SSD was trained to
	# detect, then generate a set of bounding box colors for each class
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			   "sofa", "train", "tvmonitor"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

	# load our serialized model from disk
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe("MobilenetSSD_caffe/MobileNetSSD_deploy.prototxt.txt",
								   "MobilenetSSD_caffe/MobileNetSSD_deploy.caffemodel")
	return net, CLASSES, COLORS

def LOAD_TPU_TF_MnetSSD(type, engine_int, labels_tf_int):
	# Initialize engine.
	if type == 1:
		engine_int = DetectionEngine("MobilenetSSD_tf/mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite")
		labels_tf_int = ReadLabelFile("MobilenetSSD_tf/coco_labels.txt") if "MobilenetSSD_tf/coco_labels.txt" else None
	if type == 2:
		engine_int = DetectionEngine("MobilenetSSD_tf/mobilenet_ssd_v1_voc_quant_postprocess_edgetpu.tflite")
		labels_tf_int = ReadLabelFile("MobilenetSSD_tf/voc_labels.txt") if "MobilenetSSD_tf/voc_labels.txt" else None
	if type == 3:
		engine_int = DetectionEngine("MobilenetSSD_tf/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")
		labels_tf_int = ReadLabelFile("MobilenetSSD_tf/coco_labels.txt") if "MobilenetSSD_tf/coco_labels.txt" else None
	if type == 4:
		engine_int = DetectionEngine("MobilenetSSD_tf/mobilenet_ssd_v2_voc_quant_postprocess_edgetpu.tflite")
		labels_tf_int = ReadLabelFile("MobilenetSSD_tf/voc_labels.txt") if "MobilenetSSD_tf/voc_labels.txt" else None

	return engine_int, labels_tf_int

def RUN_CPU_CAFFE_MnetSSD(frame_int, net, CLASSES, COLORS):

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame_int.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame_int, (300, 300)),
								 0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	#tracker array
	tracker_obj=0
	tracker_box_int = np.zeros([1,5])#track max 4 objects for example


	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]
		idx = int(detections[0, 0, i, 1])

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > accuracy and (idx == 6 or idx == 7 or idx == 15):
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			if(True):#
				#add to tracker
				if tracker_obj > 0:
					tracker_box_int = np.append([tracker_box_int[0]], [np.append(box, confidence)], axis=0).astype("int")
				else:
					tracker_obj+=1
					tracker_box_int[0:5] = (startX, startY, endX, endY, confidence)

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
										 confidence * 100)
			cv2.rectangle(frame_int, (startX, startY), (endX, endY),
						  (0,0, 255), 2)#COLORS[idx]
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame_int, label, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 255), 1)

	return frame_int, tracker_box_int

def RUN_TPU_TF_MnetSSD(frame_int, engine_int, labels_tf_int):
	#defs
	box_color = (255, 128, 0)
	box_thickness = 1
	label_background_color = (125, 175, 75)
	label_text_color = (255, 255, 255)
	percentage = 0.0

	#tracker array
	tracker_obj=0
	tracker_box_int = np.zeros([1,5])#track max 4 objects for example

	# Run inference.
	prepimg = frame_int[:, :, ::-1].copy()
	prepimg = Image.fromarray(prepimg)

	tinf = time.perf_counter()
	ans = engine_int.DetectWithImage(prepimg, threshold=0.5, keep_aspect_ratio=True, relative_coord=False, top_k=10)
	print(time.perf_counter() - tinf, "sec")

	# Display result.
	if ans:
		#detectframecount += 1
		for obj in ans:
			box = obj.bounding_box.flatten().tolist()
			box_left = int(box[0])
			box_top = int(box[1])
			box_right = int(box[2])
			box_bottom = int(box[3])
			cv2.rectangle(frame_int, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

			percentage = int(obj.score * 100)
			label_text = labels_tf_int[obj.label_id] + " (" + str(percentage) + "%)"

			label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
			label_left = box_left
			label_top = box_top - label_size[1]
			if (label_top < 1):
				label_top = 1
			label_right = label_left + label_size[0]
			label_bottom = label_top + label_size[1]
			cv2.rectangle(frame_int, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
						  label_background_color, -1)
			cv2.putText(frame_int, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
						label_text_color, 1)

			if(True):#TRACKER == 1
				#add to tracker
				if tracker_obj > 0:
					tracker_box_int = np.append([tracker_box_int[0]], [np.append(box[0:4], obj.score)], axis=0).astype("int")
				else:
					tracker_obj+=1
					tracker_box_int[0:5] = (label_left, label_top, label_right, label_bottom, obj.score)
	return frame_int, tracker_box_int

def FRAME_SHOW(frame_int, fps_int):
	# calc and add fps rate
	#cv2.putText(frame_int, fps_int, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (38, 0, 255), 1, cv2.LINE_AA)
	# show the output frame
	frame_int = cv2.resize(frame_int, None, fx=1, fy=1, interpolation = cv2.INTER_CUBIC)#x2 image
	cv2.imshow("Frame", frame_int)

def FPS_CHECK(stp, fps_int, framecount_int, elapsedTime_int, t1_int, t2_int):
	if(stp == 1):
		t1_int = time.perf_counter()

	if(stp == 2):
		t2_int = time.perf_counter()
		framecount_int += 1
		elapsedTime_int += t2_int - t1_int

		if elapsedTime_int > 0.3:
			fps_int = "{:.1f} FPS".format(framecount_int / elapsedTime_int)
			# print("fps = ", str(fps))
			framecount_int = 0
			elapsedTime_int = 0

	return fps_int, framecount_int, elapsedTime_int, t1_int, t2_int

def Tracker_sort(mot_tracker_int, detections, frame_int, COLORS_int):
	#run tracker
	track_bbs_ids = mot_tracker_int.update(detections).astype("int")

	#show this tracking obects
	for i in np.arange(0, track_bbs_ids.shape[0]):
		if track_bbs_ids[i, 0] > 0 and track_bbs_ids[i, 1] > 0:
			#print(track_bbs_ids[i, 4])
			(startX, startY, endX, endY, number) = track_bbs_ids[i]
			x = startX - 15 if startX - 15 > 15 else startX + 15
			cv2.putText(frame_int, str(number), (x, startY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0),2)

	return frame_int

def Tracker_opencv_init(tracker_int, frame_int, bbox_int, n_tracked_int):
	for i in range(np.size(bbox_int, 0)):
		if bbox_int[i][4] > 0.5:
			#increment - only once
			n_tracked_int += 1
			# Define an initial bounding box
			bbox = (bbox_int[i][0], bbox_int[i][1], bbox_int[i][2], bbox_int[i][3])

			# Uncomment the line below to select a different bounding box
			#bbox = cv2.selectROI(frame_int, False)

			# Initialize tracker with first frame and bounding box
			ok = tracker_int.init(frame_int, bbox)

			if ok:
				print("Tracker 2 inited!")
	return n_tracked_int

def Trackers_opencv_update(tracker_int, frame_int):
	(success_int, boxes_int) = tracker_int.update(frame_int)
	print(success_int, boxes_int)

	# loop over the bounding boxes and draw then on the frame
	if success_int:
		(x, y, w, h) = [int(v) for v in boxes_int]
		cv2.rectangle(frame_int, (x, y), (x + w, y + h), (0, 255, 0), 2)

	return success_int

def MAIN():
	width = 300
	height = 300
	fps = ""
	framecount = 0
	elapsedTime = 0
	t1 = 0
	t2 = 0

	#coffe
	net = 0
	classes = 0
	colors = 0
	stream = 0

	#TF
	pipeline = 0
	engine = 0
	labels_tf = 0

	#RS Depth
	meters=0

	#PIcamera
	if(RPI != 0):
		output = np.empty((480, 640, 3), dtype=np.uint8)

	#tracker
	if(TRACKER == 1):
		mot_tracker = Sort()

	if(TRACKER == 2):
		tracker = cv2.TrackerCSRT_create()

	tracker_box=0
	N_tracked = 0

	if(VIDEO == 0):
		if RPI == 1:
			stream = RPI_CAMERA_init((640, 480), 16)
		else:
			if RSD == 0:
				stream = USB_ONBOARD_CAMERA_init(0)
			else:
				pipeline = rs.pipeline()
				RS_D435_init(pipeline)
	else:
		stream = VIDEO_init('video_recorded/Color_2019-05-05 16:46:48.119218.avi')#'video_examples/3.mp4'

	if TPU == 0:
		net, classes, colors = LOAD_CPU_CAFFE_MnetSSD()
	else:
		engine=0
		labels_tf = 0
		engine, labels_tf = LOAD_TPU_TF_MnetSSD(4, engine, labels_tf)

	# loop over the frames from the video stream
	while(True):

		# inc FPS
		fps, framecount, elapsedTime, t1, t2 = FPS_CHECK (1, fps, framecount, elapsedTime, t1, t2)

		# Get image
		if(VIDEO == 0):
			if RPI == 1:
				frame = RPI_CAMERA_get(stream, output)
			else:
				if RSD == 0:
					frame = USB_ONBOARD_CAMERA_get(stream)
				else:
					frame, depth_image, depth_frame = RS_D435_get(pipeline)
		else:
			frame = VIDEO_get(stream)
			if frame is None:
				break

		#hold time
		t11 = time.perf_counter()
		print("t11", t11-t1)

		#FORWARD NN RUN
		if TPU == 0:
			frame, tracker_box = RUN_CPU_CAFFE_MnetSSD(frame, net, classes, colors)
		else:
			frame, tracker_box = RUN_TPU_TF_MnetSSD(frame, engine, labels_tf)

		#hold time
		t12 = time.perf_counter()
		print("t12", t12-t11)

		#tracker
		if(TRACKER == 1):
			frame = Tracker_sort(mot_tracker, tracker_box, frame, colors)

		if(TRACKER == 2):
			if N_tracked < 5:
				N_tracked = Tracker_opencv_init(tracker, frame, tracker_box, N_tracked)
			else:
				Trackers_opencv_update(tracker, frame)

		if(1):#RSD
			#Depth calculation
			print("Boxes:", tracker_box)
			if(tracker_box[0][0] != 0):
				for i in range(tracker_box.shape[0]):
					distance = (int(tracker_box[i][0] + ((tracker_box[i][2] - tracker_box[i][0]) / 2)), int(tracker_box[i][1] + (tracker_box[i][3] - tracker_box[i][1]) / 2))
					meters = 10.5#depth_frame.as_depth_frame().get_distance(distance[0], distance[1])
					cv2.putText(frame, " {:.2f}".format(meters)+" m", (distance[0], distance[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 255), 1)
				print("Distance:", meters)

		#hold time
		t13 = time.perf_counter()
		print("t13", t13-t12)

		#add text and show image on the screen
		FRAME_SHOW(frame, fps)

		#CHECK FOR KEY TO EXIT
		if (cv2.waitKey(1) & 0xFF) == ord("q"):
			break

		#hold time
		t14 = time.perf_counter()
		print("t14", t14-t13)

		print("FPS:", fps)

		# update the FPS counter
		fps, framecount, elapsedTime, t1, t2 = FPS_CHECK (2, fps, framecount, elapsedTime, t1, t2)

	# do a bit of cleanup
	cv2.destroyAllWindows()
	if(VIDEO == 0):
		if (RSD == 0):
			stream.stop()
		else:
			pipeline.stop()
	else:
		stream.release()

MAIN()

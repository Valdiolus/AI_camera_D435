#MIT license by Valdis Gerasymiak
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import math
import pyrealsense2 as rs


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
VIDEO = 1
DEPTH_VIDEO = 1
TRACKER = 0

accuracy = 0.3
# depth
depth_video_fps = 15
depth_frame_time = 1/depth_video_fps
depth_time_old=0

if(TPU != 0):
	from edgetpu.detection.engine import DetectionEngine
	from PIL import Image

if(RPI == 1):
	from picamera.array import PiRGBArray
	from picamera import PiCamera

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

def RPI_CAMERA_get(stream_int):
	output=0
	stream_int.capture(output, 'rgb')
	return output

def VIDEO_init(file, pipeline_int):
	if DEPTH_VIDEO:
		# Configure depth and color streams
		config = rs.config()

		rs.config.enable_device_from_file(config, file)

		config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
		config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

		# Start streaming
		pipeline_int.start(config)
		vs=0
	else:
		vs = cv2.VideoCapture(file)

		if (vs.isOpened() == False):
			print("Error opening video stream or file")
		depth = 0
	time.sleep(0.5)
	return vs

def VIDEO_get(stream_int):

	#frame_int = imutils.resize(frame_int, width=400)

	if DEPTH_VIDEO:
		frames = stream_int.wait_for_frames()

		color_frame = frames.get_color_frame()
		if not color_frame:
			print("No color video data!")

		depth_frame = frames.get_depth_frame()
		if not depth_frame:
			print("No depth video data!")

		frame_int = np.asanyarray(color_frame.get_data())
		depth_frame_int = np.asanyarray(depth_frame.get_data()).astype('uint8')
	else:
		ret, frame_int = stream_int.read()
		depth_frame_int=0
	return frame_int, depth_frame_int, depth_frame

def RS_D435_init(pipeline_int):
	# Configure depth and color streams
	#rs.hardware_reset()
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
	print(depth_frame)
	# Convert images to numpy arrays
	depth_image_int = np.asanyarray(depth_frame.get_data())
	color_image_int = np.asanyarray(color_frame.get_data())

	# dnn
	#im = cv2.resize(color_image_int, (300, 300))
	#im = im - 127.5
	#im = im * 0.007843q
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

def LOAD_TPU_TF_MnetSSD(type):
	# Initialize engine.
	engine_int=0
	labels_tf_int=0
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
	#1-4 = coordinates bbox, 5 - confidence, 6 - tracking number, 7 - distance
	tracker_box_int = np.zeros([10,7])#track max 10 objects for example


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
				tracker_box_int[tracker_obj, 0:5] = (startX, startY, endX, endY, confidence)
				tracker_obj += 1

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
	tracker_box_int = np.zeros([10,7])#track max 10 objects for example

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
				tracker_box_int[tracker_obj, 0:5] = (label_left, label_top, label_right, label_bottom, obj.score)
				tracker_obj+=1

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
	track_bbs_ids = mot_tracker_int.update(detections[0:5]).astype("int")

	#show this tracking obects
	for i in np.arange(0, track_bbs_ids.shape[0]):
		if track_bbs_ids[i, 0] > 0 and track_bbs_ids[i, 1] > 0:
			#print(track_bbs_ids[i, 4])
			(startX, startY, endX, endY, number) = track_bbs_ids[i]
			detections[i, 5] = number
			x = startX - 15 if startX - 15 > 15 else startX + 15
			cv2.putText(frame_int, str(number), (x, startY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0),2)

	return frame_int, detections

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

def VIDEO_INIT():
	if(VIDEO == 0):
		if RPI == 1:
			stream_int = RPI_CAMERA_init((640, 480), 16)
		else:
			if RSD == 0:
				stream_int = USB_ONBOARD_CAMERA_init(0)
			else:
				stream_int = rs.pipeline()
				RS_D435_init(stream_int)
	else:
		stream_int = rs.pipeline()
		VIDEO_init('video_recorded/Video_2019-05-14 19:59:13.603212.bag', stream_int)

	return stream_int

def DNN_INIT():
	if TPU == 0:
		net, classes, colors = LOAD_CPU_CAFFE_MnetSSD()
	else:
		colors=0
		net, classes = LOAD_TPU_TF_MnetSSD(4)
	return net, classes, colors

def TRACKER_INIT(ch):
	# tracker
	if (TRACKER):
		if ch == "sort":
			tracker_int = Sort()
		if ch == "opencv":
			tracker_int = cv2.TrackerCSRT_create()
	else:
		tracker_int = 0
	return tracker_int

def GET_FRAME(stream_int, depth_time_old_int):
	depth_image = 0
	depth_frame = 0
	if (VIDEO == 0):
		if RPI == 1:
			frame_int = RPI_CAMERA_get(stream_int)
		else:
			if RSD == 0:
				frame_int = USB_ONBOARD_CAMERA_get(stream_int)
			else:
				frame_int, depth_frame, depth_image = RS_D435_get(stream_int)
	else:
		frame_int, depth_frame, depth_image = VIDEO_get(stream_int)
		if DEPTH_VIDEO or VIDEO :
			while time.perf_counter() < (depth_time_old_int + depth_frame_time):
				a=1

	return frame_int, depth_frame, depth_image

def DNN_RUN_FORWARD(frame_int, net_int, classes_int, colors_int):
	if TPU == 0:
		frame_int, tracker_box_int = RUN_CPU_CAFFE_MnetSSD(frame_int, net_int, classes_int, colors_int)
	else:
		frame_int, tracker_box_int = RUN_TPU_TF_MnetSSD(frame_int, net_int, classes_int)

	return frame_int, tracker_box_int

def TRACKER_GO(tracker_int, tracker_box_int, frame_int, colors_int):
	# tracker
	if (TRACKER == 1):
		frame_int, tracker_box_int = Tracker_sort(tracker_int, tracker_box_int, frame_int, colors_int)

	if (TRACKER == 2):
		N_tracked=0
		if N_tracked < 5:
			N_tracked = Tracker_opencv_init(tracker_int, frame_int, tracker_box_int[0:5], N_tracked)
		else:
			Trackers_opencv_update(tracker_int, frame_int)
	return frame_int, tracker_box_int

def DEPTH_CALC(box_int, frame_int, depth_frame_int):
	if RSD or DEPTH_VIDEO:
		# Depth calculation
		#print("Boxes:", box_int)
		for i in range(box_int.shape[0]):
			if (box_int[i][4] == 0):#if confidence == 0
				break
			coordinates = np.zeros((6,2), dtype=int)
			distance = np.zeros(6)
			i_dist=0
			for k in range(-1,3,2):
				for j in range(-1,2,1):
					coordinates[i_dist][0] = (box_int[i][0] + (box_int[i][2] - box_int[i][0]) / 2)-50*j
					coordinates[i_dist][1] = (box_int[i][1] + (box_int[i][3] - box_int[i][1]) / 2)-50*k*j
					distance[i_dist] = depth_frame_int.as_depth_frame().get_distance(coordinates[i_dist][0], coordinates[i_dist][1])
					if distance[i_dist] == 0:
						distance[i_dist] = 100
					i_dist += 1
			min_dist = np.amin(distance)
			if min_dist == 0:#CHECK WHEN DISTANCE < 30 cm!!!!!!!
				print("MIN_DIST", distance)
			box_int[i, 6] = min_dist
			cv2.putText(frame_int, " {:.2f}".format(min_dist) + " m", (coordinates[1][0], coordinates[1][1]),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
	return frame_int, box_int

def DANGER_DETECTION(box_int, frame_int, saved_data_int):
	for i in range(box_int.shape[0]):
		if(box_int[i][4] == 0):#if confidence == 0
			break

		for j in range(saved_data_int.shape[0]):#SAVED DATA SATURATION!!!!!!
			if saved_data_int[j][5] == box_int[i][5]:# if captured the same ID
				cv2.putText(frame_int, "ACHTUNG", (10,10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)


	return saved_data_int, frame_int

def MAIN():
	width = 300
	height = 300
	fps = ""
	framecount = 0
	elapsedTime = 0
	t1 = 0
	t2 = 0

	tracker = TRACKER_INIT("sort")#"opencv"

	stream = VIDEO_INIT()

	net, classes, colors = DNN_INIT()

	depth_time_old = time.perf_counter()

	saved_data = np.zeros([10, 7])

	# loop over the frames from the video stream
	while(True):

		# inc FPS
		fps, framecount, elapsedTime, t1, t2 = FPS_CHECK (1, fps, framecount, elapsedTime, t1, t2)

		# Get image
		frame, depth_frame, depth_image = GET_FRAME(stream, depth_time_old)

		#depth framerate
		depth_time_old = time.perf_counter()

		#hold time
		t11 = time.perf_counter()
		print("get image", " {:.4f}".format(t11-t1))

		#FORWARD NN RUN
		frame, tracker_box = DNN_RUN_FORWARD(frame, net, classes, colors)

		#hold time
		t12 = time.perf_counter()
		print("forward NN", " {:.4f}".format(t12-t11))

		#tracker
		frame, tracker_box = TRACKER_GO(tracker, tracker_box, frame, colors)

		#hold time
		t13 = time.perf_counter()
		print("tracker", " {:.4f}".format(t13-t12))

		#Distance to objects calculation
		frame, tracker_box = DEPTH_CALC(tracker_box, frame, depth_image)

		#hold time
		t14 = time.perf_counter()
		print("depth calc", " {:.4f}".format(t14-t13))

		saved_data, frame = DANGER_DETECTION(tracker_box, frame, saved_data)

		#add text and show image on the screen
		FRAME_SHOW(frame, fps)

		#CHECK FOR KEY TO EXIT
		if (cv2.waitKey(1) & 0xFF) == ord("q"):
			break

		#hold time
		t15 = time.perf_counter()
		print("frame show", " {:.4f}".format(t15-t14))

		print("FPS:", fps)

		# update the FPS counter
		fps, framecount, elapsedTime, t1, t2 = FPS_CHECK (2, fps, framecount, elapsedTime, t1, t2)

	# do a bit of cleanup
	cv2.destroyAllWindows()
	if(VIDEO == 1):
		stream.release()
	if RSD:
		stream.stop()
	else:
		stream.stop()

MAIN()

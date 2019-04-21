#MIT license by Valdis Gerasymiak
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2

RPI = 0
RSD = 0
TPU = 0

if(RSD != 0):
	#UNCOMMENT AFTER PASTE
	import pyrealsense2 as rs

if(TPU != 0):
	from edgetpu.detection.engine import DetectionEngine
	from PIL import Image

accuracy = 0.4


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
	frame_int = imutils.resize(frame_int, width=400)
	return frame_int

def RS_D435_init(pipeline_int):
	# Configure depth and color streams
	config = rs.config()
	config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

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
	return color_image_int, color_image_int, depth_image_int #if need im or color image???

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

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > accuracy:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
										 confidence * 100)
			cv2.rectangle(frame_int, (startX, startY), (endX, endY),
						  COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame_int, label, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	return frame_int

def RUN_TPU_TF_MnetSSD(frame_int, engine_int, labels_tf_int):
	#defs
	box_color = (255, 128, 0)
	box_thickness = 1
	label_background_color = (125, 175, 75)
	label_text_color = (255, 255, 255)
	percentage = 0.0

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
			cv2.putText(frame_int, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
						label_text_color, 1)
	return frame_int

def FRAME_SHOW(frame_int, fps_int):
	# calc and add fps rate
	cv2.putText(frame_int, fps_int, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (38, 0, 255), 1, cv2.LINE_AA)
	# show the output frame
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

	if RSD == 0:
		stream = USB_ONBOARD_CAMERA_init(0)
	else:
		pipeline = rs.pipeline()
		RS_D435_init(pipeline)

	if TPU == 0:
		net, classes, colors = LOAD_CPU_CAFFE_MnetSSD()
	else:
		engine=0
		labels_tf = 0
		engine, labels_tf = LOAD_TPU_TF_MnetSSD(0, engine, labels_tf)

	# loop over the frames from the video stream
	while(True):

		# inc FPS
		fps, framecount, elapsedTime, t1, t2 = FPS_CHECK (1, fps, framecount, elapsedTime, t1, t2)

		#Get image
		if RSD == 0:
			frame = USB_ONBOARD_CAMERA_get(stream)
		else:
			frame, color_image, depth_image = RS_D435_get(pipeline)

		#FORWARD NN RUN
		if TPU == 0:
			frame = RUN_CPU_CAFFE_MnetSSD(frame, net, classes, colors)
		else:
			frame = RUN_TPU_TF_MnetSSD(frame, engine, labels_tf)

		#add text and show image on the screen
		FRAME_SHOW(frame, fps)

		#CHECK FOR KEY TO EXIT
		if (cv2.waitKey(1) & 0xFF) == ord("q"):
			break

		# update the FPS counter
		fps, framecount, elapsedTime, t1, t2 = FPS_CHECK (2, fps, framecount, elapsedTime, t1, t2)

	# do a bit of cleanup
	cv2.destroyAllWindows()
	if (RSD == 0):
		stream.stop()
	else:
		pipeline.stop()

MAIN()
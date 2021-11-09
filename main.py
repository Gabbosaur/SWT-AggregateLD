import cv2
import os
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg", force=True)
#%matplotlib inline
#plt.show()

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
	'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
	'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
	'APIMODEL_PATH': os.path.join('Tensorflow','models'),
	'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
	'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
	'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
	'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
	'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
	'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
	'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
	'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
	'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
	'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
	'COCO_NAMES':os.path.join('Tensorflow', 'workspace','pre-trained-models', 'yolo_V3' , 'coco.names'),
	'YOLOV3_CFG':os.path.join('Tensorflow', 'workspace','pre-trained-models', 'yolo_V3' , 'yolov3.cfg'),
	'YOLOV3_SPP_WEIGHTS':os.path.join('Tensorflow', 'workspace','pre-trained-models', 'yolo_V3' , 'yolov3.weights'),
	'PRETRAINED_MODEL':os.path.join('Tensorflow', 'workspace','pre-trained-models', PRETRAINED_MODEL_NAME,'saved_model'),
	'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
	'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

#######################################YOLO V3, non utilizzata perché rileva oggetti 
def testConYoloV3():
	# Load Yolo
	print("LOADING YOLO")
	net = cv2.dnn.readNet(files['YOLOV3_SPP_WEIGHTS'], files['YOLOV3_CFG'])
	#save all the names in file o the list classes
	classes = []
	with open(files['COCO_NAMES'], "r") as f:
		classes = [line.strip() for line in f.readlines()]

	print(classes)

	#get layers of the network
	layer_names = net.getLayerNames()
	#Determine the output layer names from the YOLO model
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	print("YOLO LOADED")


	# Capture frame-by-frame
	IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'thumbsdown.b1f20c56-b4d4-11eb-ae88-240a64b78789.jpg')
	#IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'thumbsup.fd7cdb14-b4d4-11eb-b662-240a64b78789.jpg')
	img = cv2.imread(IMAGE_PATH)
	#img=cv2.imread("test_img.jpg")
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape

	# USing blob function of opencv to preprocess image
	#blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),swapRB=True, crop=False)

	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)

	#for b in blob:
	#	for n,img_blob in enumerate(b):
	#		cv2.imshow(str(n),img_blob)

	#Detecting objects
	net.setInput(blob)
	outs = net.forward(output_layers)


	# Showing informations on the screen
	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.3:
				# Object detected
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)

				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)

				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)


	isPerson=False
	#We use NMS function in opencv to perform Non-maximum Suppression
	#we give it score threshold and nms threshold as arguments.
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			if(label=="person"):
				isPerson=True
			color = colors[class_ids[i]]
			cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
			cv2.putText(img, label, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX,1/2, color, 2)

	cv2.imshow("Image",img)
	cv2.waitKey(0)


import yolov5
import torch

# YOLOV5_model=os.path.join('tfod', 'Lib','site-packages', 'yolov5' ,'models', 'yolov5s.yaml'),
# print(os.getcwd()) # returns the currently working directory of a process
# model = yolov5.load(YOLOV5_model) # carico modello tramite formato yaml, però non funziona
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) # scarico il modello
model = yolov5.load('yolov5s.pt') # se ho il modello in locale

def rilevaPersona(model, frame):
	#######################################YOLO V5

	isPerson=False
	## Immagine per test rilevazione persona
	# IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'livelong.fed895dc-b264-11eb-bccc-086266b476b9.jpg')

	# img = cv2.imread(frame)


	# inference
	#results = model(img)

	results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size=400)

	labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy() # estraggo labels e coordinate dei rettangoli
	classes = model.names
	# show results
	for i in labels:
		# print(classes[int(i)])
		if(classes[int(i)]=="person"):
			isPerson=True

	# show results
	# results.print()
	# results.show()


	# Load pipeline config and build a detection model
	configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
	detection_model = model_builder.build(model_config=configs['model'], is_training=False)

	@tf.function
	def detect_fn(image):
		image, shapes = detection_model.preprocess(image)
		prediction_dict = detection_model.predict(image, shapes)
		detections = detection_model.postprocess(prediction_dict, shapes)
		return detections


	if(isPerson==True):
		print("PERSONA RICONOSCIUTA")
		category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
		#IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'thumbsdown.b1f20c56-b4d4-11eb-ae88-240a64b78789.jpg')
		#IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'prova_scritte.jpg')

		# img = cv2.imread(frame)
		image_np = np.array(frame)

		input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
		detections = detect_fn(input_tensor)

		num_detections = int(detections.pop('num_detections'))
		detections = {key: value[0, :num_detections].numpy()
					for key, value in detections.items()}
		detections['num_detections'] = num_detections

		# detection_classes should be ints.
		detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

		label_id_offset = 1
		image_np_with_detections = image_np.copy()

		viz_utils.visualize_boxes_and_labels_on_image_array(
					image_np_with_detections,
					detections['detection_boxes'],
					detections['detection_classes']+label_id_offset,
					detections['detection_scores'],
					category_index,
					use_normalized_coordinates=True,
					max_boxes_to_draw=5,
					min_score_thresh=.8,
					agnostic_mode=False)

		# plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
		# plt.savefig("img_with_person.png")
		# plt.show()
	else:
		print("PERSONA NON RICONOSCIUTA")


	return isPerson



###frame from video with words detection
import pytesseract
pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
import cv2
import mediapipe as mp
from deepface import DeepFace
from dataclasses import dataclass

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

width, height, padding = 0, 0, 50

@dataclass
class Record: # 1 frame
	num_frame: int
	time: float
	list_words: list
	isPersonDetected: bool
	# isSpeaker: bool
	# path_faces: list 			# ['faccia1', 'faccia2', 'faccia3']
	# idFaces: list 			# [0,1,2] in questo caso ci sono 3 persone distinte e alla fine contare le occorrenze e scegliere il massimo, se il numero delle occorrenze fossero uguali, si prende quello con l'id più basso

# id face, id che assegneremo al momento del compare

@dataclass
class FrameWithFaces:
	num_frame: int
	num_faces: int


listOfRecords=[]
IMAGE_FACES_PATH = 'images/faces/'
NOME_VIDEO = 'prova.mp4'
video = cv2.VideoCapture(NOME_VIDEO)
i = 0
# a variable to set how many frames you want to skip
fps = video.get(cv2.CAP_PROP_FPS)
frame_skip = fps*5 #un frame ogni 5 secondi
frame_counter=0
temp=1/fps
n_frame_analyzed = 0
frame_with_faces = []

count_frame_doppi=0

while video.isOpened():
	ret, frame = video.read()
	if not ret:
		break
	if i > frame_skip - 1: # In questo caso ogni 5 secondi
		n_frame_analyzed+=1
		#cv2.imwrite('test_'+str(i)+'.jpg', frame)
		#print(pytesseract.image_to_string(frame))
		temp_list_words=[]
		boxes=pytesseract.image_to_data(frame)
		frame_counter+=frame_skip
		#print("frame number: " + str(frame_counter))

		# Riconoscimento persona
		isPersonDetected = rilevaPersona(model, frame)

		if isPersonDetected == True:
			with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:

				# image = cv2.imread(file)
				print('width: ', frame.shape[1])
				print('height:', frame.shape[0])
				width = frame.shape[1]
				height = frame.shape[0]

				# Convert the BGR image to RGB and process it with MediaPipe Face Detection.
				results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

				# Draw face detections of each face.
				if not results.detections:
					print("Nessuna faccia rilevata.")
					continue

				# annotated_image = image.copy()

				# Popola la lista con un frame con tot facce, quando faremo il compare delle facce, useremo questa lista per ottimizzare i tempi
				frame_with_faces.append(FrameWithFaces(n_frame_analyzed, len(results.detections)))

				for i in range(len(results.detections)):
					detection = results.detections[i]
					print("Faccia rilevata.")
					# print(detection.location_data.relative_bounding_box)
					xmin = int(detection.location_data.relative_bounding_box.xmin*width)
					ymin = int(detection.location_data.relative_bounding_box.ymin*height)
					w = int(detection.location_data.relative_bounding_box.width*width)+xmin
					h = int(detection.location_data.relative_bounding_box.height*height)+ymin

					# controllo immagine croppata che resti dentro all'immagine originale
					if xmin - padding >= 0:
						xmin = xmin - padding
					else:
						xmin = 0

					if ymin - padding >= 0:
						ymin = ymin - padding
					else:
						ymin = 0

					if w + padding <= width:
						w = w + padding
					else:
						w = width

					if h + padding <= height:
						h = h + padding
					else:
						h = height


					print(xmin)
					print(ymin)
					print(w)
					print(h)

					# print('Nose tip:')
					# print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
					cropped_image = frame[ymin:h, xmin:w]
					cv2.imwrite(IMAGE_FACES_PATH + 'face_of_' + str(NOME_VIDEO[:-4]) + '_frame_'+ str(n_frame_analyzed) + '_face_' + str(i) + '.jpg', cropped_image)

					cv2.waitKey(0)
					cv2.destroyAllWindows()





		for x,b in enumerate(boxes.splitlines()):
			if x!=0:
				b=b.split()
				if len(b)==12: # gli indici prima del 12 sono altre informazioni del testo (colore, posizione, etc..)
					#print(b[11])
					#lista delle parole
					temp_list_words.append(b[11])
		#struct con numero del frame e la lista di parole
		rec=Record(frame_counter,frame_counter*temp,temp_list_words, isPersonDetected)
		#controlliamo se questo identico record è già contenuto nella lista
		#potrebbe essere dispendioso, facciamo solo il compare con l'ultimo frame in listOfRecords?
		flag=False

		# conta se ci sono frames con la lista delle parole uguali
		for y in listOfRecords:
			if(y.list_words == rec.list_words):
				flag=True
				count_frame_doppi += 1
				break

		if(flag==False):
			listOfRecords.append(rec) # avremo listOfRecords UNIVOCI
		i = 0
		continue
	i += 1

video.release()
cv2.destroyAllWindows()



# SVILUPPARE IL CICLO CHE VA A COMPARARE TUTTE LE FACCE
# Compara le due foto e definisce se le due persone trovate sono la stessa persona
# result = DeepFace.verify(img1_path = "images/lec3.jpg", img2_path = IMAGE_FACES_PATH+"face_of_lec2_face0.jpg")
# print(result)
print(frame_with_faces)



#print(listOfRecords)
for y in listOfRecords:
	duration=y.time
	minutes = int(duration/60)
	seconds = int(duration%60)
	print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))
	print(y.list_words)
	print("Persona presente? " + str(y.isPersonDetected))
	print("\n")
#print("frame doppi:"+ str(count_frame_doppi))
#print(listOfRecords[0])



# 0. Setup Paths


'''
problemi:
	come usare variabili all'interno dei path
'''
import os

#os.system('cd C:\\Users\\luca\\Desktop\\tensorflow_object_detaction\\TFODCourse')
#os.system('.\tfod\Scripts\activate')


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
'''
for path in paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            os.system('mkdir -p {path}')
        if os.name == 'nt':
            os.system('mkdir {path}')
'''
# 1. Download TF Models Pretrained Models from Tensorflow Model Zoo and Install TFOD

# https://www.tensorflow.org/install/source_windows

if os.name=='nt':
    #os.system('pip install wget')
    import wget




'''
if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    os.system('git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}')
'''
'''
# Install Tensorflow Object Detection 
if os.name=='posix':  
    os.system('apt-get install protobuf-compiler')
    os.system('cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install . ')
    
if os.name=='nt':
    url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
    wget.download(url)
    os.system('move protoc-3.15.6-win64.zip {paths['PROTOC_PATH']}')
    os.system('cd {paths['PROTOC_PATH']} && tar -xf protoc-3.15.6-win64.zip')
    os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))   
    os.system('cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install')
    os.system('cd Tensorflow/models/research/slim && pip install -e . ')

VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
# Verify Installation
os.system('python {VERIFICATION_SCRIPT}')

os.system('pip install protobuf')

os.system('pip install tensorflow --upgrade')

os.system('pip uninstall protobuf matplotlib -y')
os.system('pip install protobuf matplotlib==3.2')

os.system('pip install pyyaml')
'''
import object_detection

'''
if os.name =='posix':
    os.system('wget {PRETRAINED_MODEL_URL}')
    os.system('mv {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}')
    os.system('cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}')
if os.name == 'nt':
    wget.download(PRETRAINED_MODEL_URL)
    os.system('move {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}')
    os.system('cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}')
'''
# 2. Create Label Map

labels = [{'name':'ThumbsUp', 'id':1}, {'name':'ThumbsDown', 'id':2}, {'name':'ThankYou', 'id':3}, {'name':'LiveLong', 'id':4}]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

# 3. Create TF records

# OPTIONAL IF RUNNING ON COLAB
ARCHIVE_FILES = os.path.join(paths['IMAGE_PATH'], 'archive.tar.gz')
if os.path.exists(ARCHIVE_FILES):
  os.system('tar -zxvf {'+ARCHIVE_FILES+'}')

if not os.path.exists(files['TF_RECORD_SCRIPT']):
    os.system('git clone https://github.com/nicknochnack/GenerateTFRecord {'+paths['SCRIPTS_PATH']+'}')

command="python {} -x {} -l {} -o {}".format(files['TF_RECORD_SCRIPT'],os.path.join(paths['IMAGE_PATH'], 'train'),files['LABELMAP'],os.path.join(paths['ANNOTATION_PATH'], 'train.record'))
os.system(command)

command="python {} -x {} -l {} -o {}".format(files['TF_RECORD_SCRIPT'],os.path.join(paths['IMAGE_PATH'], 'test'),files['LABELMAP'],os.path.join(paths['ANNOTATION_PATH'], 'test.record'))
os.system(command)

# 4. Copy Model Config to Training Folder

if os.name =='posix':
	os.system('cp {os.path.join('+ paths['PRETRAINED_MODEL_PATH']+', '+PRETRAINED_MODEL_NAME+', \'pipeline.config\')} {os.path.join('+paths['CHECKPOINT_PATH']+')}')
if os.name == 'nt':
	command="copy {} {}".format(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME,'pipeline.config'),os.path.join(paths['CHECKPOINT_PATH']))
	os.system(command)

# 5. Update Config For Transfer Learning

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format



config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])


pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
	    f.write(config_text)

# 6. Train the model
'''
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')

command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=2000".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])

!pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1


!{command}
'''

# 7. Evaluate the Model
'''
command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])

print(command)

!{command}

'''

# 8. Load Train Model From Checkpoint

import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

paths['CHECKPOINT_PATH']


##########################

#pretrained_model = tf.saved_model.load(files['PRETRAINED_MODEL'])




# 9. Detect from an Image

import cv2 
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
#%matplotlib inline
plt.show()
'''
#######################################TOLO V3
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
'''
#######################################YOLO V5
isPerson=False

import yolov5
import torch

IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'thumbsdown.b1f20c56-b4d4-11eb-ae88-240a64b78789.jpg')

# YOLOV5_model=os.path.join('tfod', 'Lib','site-packages', 'yolov5' ,'models', 'yolov5s.yaml'),
print(os.getcwd())
# model = yolov5.load(YOLOV5_model) # carico modello tramite formato yaml, però non funziona
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) # scarico il modello
model = yolov5.load('yolov5s.pt') # se ho il modello in locale

img = cv2.imread(IMAGE_PATH)


# inference
#results = model(img)

results = model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), size=400)

labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy() # estraggo labels e coordinate dei rettangoli
classes = model.names
# show results
for i in labels:
	print(classes[int(i)])
	if(classes[int(i)]=="person"):
		isPerson=True

# show results
results.print()
results.show()



if(isPerson==True):
	print("PERSONA RICONOSCIUTA")
	category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
	#IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'thumbsdown.b1f20c56-b4d4-11eb-ae88-240a64b78789.jpg')
	#IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'prova_scritte.jpg')

	img = cv2.imread(IMAGE_PATH)
	image_np = np.array(img)

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

	plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
	plt.show()

else:
	print("PERSONA NON RICONOSCIUTA")






###frame from video with words detection
import pytesseract
pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


from dataclasses import dataclass

@dataclass
class Record:
    num_frame: int
    time: float
    list_words: list

listOfRecords=[]
video = cv2.VideoCapture('prova.mp4')
i = 0
# a variable to set how many frames you want to skip
fps = video.get(cv2.CAP_PROP_FPS)
frame_skip = fps*5 #un frame ogni 5 secondi
frame_counter=0
temp=1/fps

count_frame_doppi=0

while video.isOpened():
	ret, frame = video.read()
	if not ret:
		break
	if i > frame_skip - 1:
		#cv2.imwrite('test_'+str(i)+'.jpg', frame)
		#print(pytesseract.image_to_string(frame))
		temp_list_words=[]
		boxes=pytesseract.image_to_data(frame)
		frame_counter+=frame_skip
		#print("frame number: " + str(frame_counter))
		for x,b in enumerate(boxes.splitlines()):
			if x!=0:
				b=b.split()
				if len(b)==12:
					#print(b[11])
					#lista delle parole
					temp_list_words.append(b[11])
		#struct con numero del frame e la lista di parole
		rec=Record(frame_counter,frame_counter*temp,temp_list_words)
		#controlliamo se questo identico record è già contenuto nella lista
		#potrebbe essere dispendioso, facciamo solo il compare con l'ultimo frame in listOfRecords?
		flag=False
		for y in listOfRecords:
			if(y.list_words == rec.list_words):
				flag=True
				count_frame_doppi += 1
				break

		if(flag==False):
			listOfRecords.append(rec)
		i = 0
		continue
	i += 1

video.release()
cv2.destroyAllWindows()

#print(listOfRecords)
for y in listOfRecords:
	duration=y.time
	minutes = int(duration/60)
	seconds = int(duration%60)
	print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))
	print(y.list_words)

#print("frame doppi:"+ str(count_frame_doppi))
#print(listOfRecords[0])

# 10. Real Time Detections from your Webcam

os.system('pip uninstall opencv-python-headless -y')

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened(): 
    ret, frame = cap.read()
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

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

# 10. Freezing the Graph

FREEZE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'exporter_main_v2.py ')

command = "python {} --input_type=image_tensor --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(FREEZE_SCRIPT ,files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'], paths['OUTPUT_PATH'])
os.system(command)
#print(command)
'''
!{command}
'''
# 11. Conversion to TFJS

os.system('pip install tensorflowjs')

command = "tensorflowjs_converter --input_format=tf_saved_model --output_node_names='detection_boxes,detection_classes,detection_features,detection_multiclass_scores,detection_scores,num_detections,raw_detection_boxes,raw_detection_scores' --output_format=tfjs_graph_model --signature_name=serving_default {} {}".format(os.path.join(paths['OUTPUT_PATH'], 'saved_model'), paths['TFJS_PATH'])
os.system(command)

#print(command)
'''
!{command}
'''
# Test Code: https://github.com/nicknochnack/RealTimeSignLanguageDetectionwithTFJS

# 12. Conversion to TFLite

TFLITE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'export_tflite_graph_tf2.py ')

command = "python {} --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(TFLITE_SCRIPT ,files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'], paths['TFLITE_PATH'])
os.system(command)
#print(command)
'''
!{command}
'''

FROZEN_TFLITE_PATH = os.path.join(paths['TFLITE_PATH'], 'saved_model')
TFLITE_MODEL = os.path.join(paths['TFLITE_PATH'], 'saved_model', 'detect.tflite')

command = "tflite_convert \
--saved_model_dir={} \
--output_file={} \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=FLOAT \
--allow_custom_ops".format(FROZEN_TFLITE_PATH, TFLITE_MODEL, )

os.system(command)

#print(command)
'''
!{command}
'''
# 13. Zip and Export Models 

os.system('tar -czf models.tar.gz {paths[\'CHECKPOINT_PATH\']}')
'''
from google.colab import drive
drive.mount('/content/drive')
'''
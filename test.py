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

#######################################


import yolov5
import torch
IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'thumbsdown.b1f20c56-b4d4-11eb-ae88-240a64b78789.jpg')

YOLOV5_model=os.path.join('tfod', 'Lib','site-packages', 'yolov5' ,'models', 'yolov5s.yaml'),
#model
print(os.getcwd())
#model = yolov5.load(YOLOV5_model)
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = yolov5.load('yolov5s.pt')

img = cv2.imread(IMAGE_PATH)
#img=cv2.imread("test_img.jpg")

# inference
#results = model(img)

results = model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), size=400)

labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
classes = model.names
# show results
for i in labels:
	print(classes[int(i)])
	if(classes[int(i)]=="person"):
		isPerson=True

results.print()

results.show()


'''
command="python {} --source {}".format(os.path.join("tfod","Lib","site-packages","yolov5",'detect.py'),"dublin.mp4")
os.system(command)
'''
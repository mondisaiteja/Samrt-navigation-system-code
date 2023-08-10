# Samrt-navigation-system-code
It is my final year project, which is a smart navigation system for visually impaired people using computer vision algorithms and IOT devices.

First to use this code we need to download the required object detection algorithm.
At the position of model name please uncomment the lines which are commented below and add at the respected place, then the algorithm can be downloaded and extracted automatically.
It is better to do in jupyter.
=====================================================================================
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
# MODEL_NAME = 'faster_rcnn_resnet101_coco_2017_11_08'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 50
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())
======================================================================================

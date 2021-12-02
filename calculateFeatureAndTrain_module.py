import numpy as np
import cv2 as cv
import pathlib
import os
from matplotlib import pyplot as plt


def plotList(array):
	objects = ('0', '1', '2', '3', '4', '5','6','7','8','9','10','11','12','13','14','15')
	y_pos = np.arange(len(objects))
	performance = array

	plt.bar(y_pos, performance, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)

	plt.show()


PROJECT_PATH=pathlib.Path(__file__).parent.resolve() #restituisce il path del progetto

IMAGE_TRAIN_PATH= "images\\train"
FINAL_IMAGE_TRAIN_PATH=os.path.join(PROJECT_PATH,IMAGE_TRAIN_PATH)

list_subfolders_with_paths = [f.path for f in os.scandir(FINAL_IMAGE_TRAIN_PATH) if f.is_dir()]

numero_gruppi = 16


X_train = []
y_train = []

for directory in list_subfolders_with_paths:
	
	#cancella le 2 righe sotto al commento
	#cartella=os.path.basename(os.path.normpath(directory))
	#if cartella=="noAlzateLaterali":
		#directory=FULL_VIDEO_PATH
	print("sto processando le img in: "+ directory )
	for file in os.listdir(directory):
		path_directory = os.path.join(FINAL_IMAGE_TRAIN_PATH, directory)
		filename = os.fsdecode(file)
		final_file_path = os.path.join(path_directory, filename)

		img = cv.imread(final_file_path,0)
		hist = cv.calcHist([img],[0],None,[256],[0,256])

		dimensione_gruppo = len(hist) / numero_gruppi

		counter = 0
		array_of_hist = []
		sum = 0
		for i in range(0, len(hist)):
			counter+=1

			sum = sum + hist[i][0].astype(int)
			if counter == dimensione_gruppo:
				array_of_hist.append(sum)
				counter = 0
				sum = 0

		totalPixel = np.sum(array_of_hist)

		normalized_array_of_hist = [el / totalPixel for el in array_of_hist]

		# plotList(normalized_array_of_hist)
		X_train.append(normalized_array_of_hist)

		if os.path.basename(directory) == 'blackboard':
			y_train.append(0)
		elif os.path.basename(directory) == 'slide':
			y_train.append(1)
		elif os.path.basename(directory) == 'slide-and-talk':
			y_train.append(2)
		elif os.path.basename(directory) == 'talk':
			y_train.append(3)

print("X_train:")
print(len(X_train))

print("y train:")
print(len(y_train))

# y: 0 blackboard, 1 slide, 2 slide n talk, 3 talk

# cercare più immagini per il dataset
# aggiungere la faccia in X_train (boolean)
# dare in pasto a SVM (TENERE un paio di immagini per il TEST)

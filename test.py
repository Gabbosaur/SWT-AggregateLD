import cv2
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

print("frame doppi:"+ str(count_frame_doppi))
#print(listOfRecords[0])


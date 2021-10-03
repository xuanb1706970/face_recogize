import cv2, os
import numpy as np
import time
from array_label import  data, data_update, data_name, array_data_haar, array_data_dlib
from detec_face import video_predict


recognizer = cv2.face.LBPHFaceRecognizer_create()

#Update mô hình
def update():
	print('update')
	faces, labels = data_update()

	# face_recognizer = cv2.face.createLBPHFaceRecognizer()
	start = time.time()
	recognizer.update(faces, np.array(labels))
	print("--- %s seconds ---" % (time.time() - start))
	recognizer.save('trainningData.yml')
	data_name()

#Train mô hình
def train():
	print('train')
	start = time.time()
	faces, labels = array_data_haar()

	print(set(labels))
	# face_recognizer = cv2.face.createLBPHFaceRecognizer()
	recognizer.train(faces, np.array(labels))
	recognizer.save('trainningData_haar.yml')
	end = time.time()
	print("TRAIN")
	print("Thoi gian train mo hinh la: ", end - start)

def key_work(x):
	if x == 1:
		train()
	elif x == 2:
		update()
	elif x == 3:
		video_predict()
	else:
		print("Not is Number")

if __name__ == "__main__":
	num = input("1: Train\n2: Update\n3: Video_Predict\nPlease enter number:")
	key_work(int(num))
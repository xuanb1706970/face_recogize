import csv

import cv2, os
import dlib
from array_label import detect_face

def read_csv():
	data_name = []
	with open ('data_name_test.csv', 'r') as f:
		read_name = csv.reader(f)
		for row in read_name:
			for e in row:
				data_name.append(e)
	return data_name

#video_capture = cv2.VideoCapture(1)
def video_predict():
	video_capture = cv2.VideoCapture(0)
	path = '../face_recogize/trainningData.yml'
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	face_recognizer.read(path)
	dertector = dlib.get_frontal_face_detector()

	dataset = read_csv()

	while True:
		ret, frame = video_capture.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = dertector(gray)
		for face in faces:
			x1 = face.left()
			y1 = face.top()
			x2 = face.right()
			y2 = face.bottom()

			IDs, conf = face_recognizer.predict(gray[y1:y2,x1:x2])
			labels_text = dataset[IDs]

			print(IDs)

			cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),3)
			cv2.putText(frame, labels_text, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
			cv2.imshow('image',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	# img = cv2.imread('Dataset/1/1.jpg')
	# img = predict(img)
	# cv2.imshow('Image', img)
	# cv2.waitKey(1000)
	video_capture.release()
	cv2.destroyAllWindows()

#predict image Data_Test (Hinh anh duoc su dung dlib cat)
def detect_image():
	path = '../face_recogize/trainningData_dlib.yml'
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	face_recognizer.read(path)
	count = 0
	for i in os.listdir('Data_Test_Cut'):
		path = 'Data_Test_Cut/'+i
		print(path)
		ID = int(os.path.split(path)[-1].split('.')[0])
		img = cv2.imread(path)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		IDs, conf = face_recognizer.predict(gray)

		print("ID Predict: ", IDs)
		print("ID Image: ", ID)
		if int(IDs) == ID:
			count += 1
			print(count)
		else:
			print("None")

	return count

#predict image Data_Test  (hinh anh nguyen ban)
def detect_image_dlib():
	detector = dlib.get_frontal_face_detector()
	path = '../face_recogize/trainningData_dlib.yml'
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	face_recognizer.read(path)
	count = 0

	for i in os.listdir('Data_Test'):
		path = 'Data_Test/'+i
		ID = int(os.path.split(path)[-1].split('.')[0])
		img = cv2.imread(path)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = detector(gray)
		if len(faces) == 0 or len(faces) > 2:
			print("None")
		else:
			for face in faces:
				x1 = face.left()
				y1 = face.top()
				x2 = face.right()
				y2 = face.bottom()

				IDs, conf = face_recognizer.predict(gray[y1:y2, x1:x2])

				print("ID Predict: ", IDs)
				print("ID Image: ", ID)
				if int(IDs) == ID:
					count += 1
					print(count)
				else:
					print("None")
	return count

#Tinh phan tram của haar (hinh anh nguyen ban)
def detect_image_haar():
	path_harr = 'haarcascade_frontalface_default.xml'
	path = '../face_recogize/trainningData_haar.yml'
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	face_recognizer.read(path)
	count = 0

	for i in os.listdir('Data_Test'):
		path = 'Data_Test/'+i
		ID = int(os.path.split(path)[-1].split('.')[0])
		img = cv2.imread(path)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faceCasc = cv2.CascadeClassifier(path_harr)
		faces = faceCasc.detectMultiScale(gray, 1.3, 5)

		if len(faces) == 0 or len(faces) > 1:
			print('None')
		else:
			(x, y, w, h) = faces[0]

			IDs , faces = face_recognizer.predict(gray[y:y + w, x:x + h])

			print("ID Predict: ", IDs)
			print("ID Image: ", ID)
			if int(IDs) == ID:
				count += 1
				print(count)
			else:
				print("None")
	return count

#Tinh phan tram
def Tong():
	count = detect_image_haar()
	print("Detec_Image Haar")
	print("{:.0%}".format(count/500))


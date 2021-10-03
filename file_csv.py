import os
import csv
import dlib
import cv2
from PIL import Image

path = 'D:\image\lfw'
import numpy

#Doi ten thanh number
def change_name():
    path_floder = 'D:\image\data'
    for count, file in enumerate(os.listdir(path_floder), start=1):
        os.rename(os.path.join(path_floder, file), os.path.join(path_floder, str(count)))

#Ghi ten vao file csv
def write_csv():
    data = []
    for i in os.listdir(path):
        data.append(i)

    with open('data_name_test.csv', 'w+') as f:
        write = csv.writer(f)
        write.writerow({""})
        for j in data:
            write.writerow({j})

#Doc file csv
def read_csv():
    data_name = []
    with open('data_name_test.csv', 'r') as f:
        data_name.append(f.read())

    for i in data_name:
        print(i)

#Chuẩn bị hình ảnh cắt bằng dlib để test
def data_image():
    detector = dlib.get_frontal_face_detector()
    number = 1
    number_image = 1
    for i in os.listdir("Data_Root"):
        floder = "Data_Root/" + i

        for j in os.listdir(floder):
            if number_image <= 1:
                if number < 501:
                    path = floder + '/' + j
                    img = cv2.imread(path)
                    img = cv2.resize(img, (224,224))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray)
                    if len(faces) == 0 or len(faces) > 1:
                        print('None')
                    else:
                        cv2.imwrite(os.path.join('Data_Test/' + str(i) + '.' + j), img)
                        number += 1
                        number_image += 1
                else:
                    break
            else:
                number_image = 1
                break

#Chuan bi tap du lieu Test đã cắt
def data_image_cut():
    data_name = []
    number_image = 1
    number = 1
    predictor_path ='shape_predictor_5_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()

    for i in os.listdir("Data_Root"):
        floder = "Data_Root/" + i

        for j in os.listdir(floder):
            if number_image <= 1:
                if number < 501:
                    path = floder + '/' + j
                    sp = dlib.shape_predictor(predictor_path)

                    img = dlib.load_rgb_image(path)

                    dets = detector(img)

                    if len(dets) == 0 or len(dets) > 1:
                        print("None")
                    else:
                        faces = dlib.full_object_detections()
                        for detection in dets:
                            faces.append(sp(img, detection))

                        image = dlib.get_face_chip(img, faces[0], size=320)
                        jittered_images = dlib.jitter_image(image, num_jitters=1)
                        faces_gray = []
                        for img in jittered_images:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            faces_gray.append(gray)

                        for t1 in faces_gray:
                            cv2.imwrite(os.path.join('Data_Test_Cut/' + str(i) + '.' + j), t1)
                            number += 1
                            number_image += 1
                else:
                    break
            else:
                number_image = 1
                break

#Chuân bị tập dữ liệu cho dlib để train
def handle_image_dlib():
    data_name = []
    predictor_path = 'shape_predictor_5_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()

    for i in os.listdir("Data_Root"):
        floder = "Data_Root/" + i

        for j in os.listdir(floder):

            path = floder + '/' + j
            sp = dlib.shape_predictor(predictor_path)

            img = dlib.load_rgb_image(path)

            dets = detector(img)

            if len(dets) != 0 and len(dets) < 2:

                faces = dlib.full_object_detections()
                for detection in dets:
                    faces.append(sp(img, detection))

                    image = dlib.get_face_chip(img, faces[0], size=320)
                    jittered_images = dlib.jitter_image(image, num_jitters=1)
                    faces_gray = []
                    for img in jittered_images:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces_gray.append(gray)

                    for t1 in faces_gray:

                        if os.path.exists(os.path.join('Data_Dlib/', str(i))) == False:
                            os.mkdir(os.path.join('Data_Dlib/', str(i)))

                        cv2.imwrite(os.path.join('Data_Dlib/' + str(i) + '/' + str(i) + '.' + j), t1)
            else:
                break


#Chuẩn bị dữ liệu hình ảnh để train bằng haar_like
def harr_like_image():
    path_harr = 'haarcascade_frontalface_default.xml'
    for i in os.listdir("Data_Root"):
        floder = "Data_Root/" + i
        for j in os.listdir(floder):
            path = floder + '/' + j
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faceCasc = cv2.CascadeClassifier(path_harr)
            faces = faceCasc.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0 or len(faces) > 1:
                print('None')
            else:
                (x, y, w, h) = faces[0]
                if os.path.exists(os.path.join('Data_Haar/', str(i))) == False:
                    os.mkdir(os.path.join('Data_Haar/', str(i)))

                cv2.imwrite(os.path.join('Data_Haar/' + str(i) + '/' + str(i) + '.' + j), gray[y:y + w, x:x + h])


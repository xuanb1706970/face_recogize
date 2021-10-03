import  cv2
import os
import csv
from PIL import Image
import  numpy as np
import dlib

#Cat hinh anh va train bang harrcades
def detect_face(img) :
    path = 'haarcascade_frontalface_default.xml'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCasc = cv2.CascadeClassifier(path)
    faces = faceCasc.detectMultiScale(gray, 1.3, 5)
    graylist = []
    faceslist = []

    if len(faces) == 0:
        return None, None

    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0]

#array: Faces and Labels Train
def data():
    dirs = os.listdir("Data_Root")

    faces = []
    labels = []

    for i in dirs:
        set = "../face_recogize/Data_Root/" + i
        label = int(i)
        for j in os.listdir(set):
            path = set + "/" + j
            img = cv2.imread(path)
            face, ret = detect_face(img)

            if face is not None:
                faces.append(face)

                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    print('Mang number: ', labels)
    return faces, labels

#array: Faces and Labels Update
def data_update():
    dirs = os.listdir("Data_Test_Cut")

    check = 0
    data_name = []

    with open('labels_data.csv', 'r') as f:
        read = f.read()
        read = read.strip('\n')
        for i in (read.split(',')):
            data_name.append(i)

    faces = []
    labels = []
    for i in dirs:
        set_floder = "../face_recogize/Data_test/" + i
        for k in set(data_name):
            if int(k) == int(i):
                check = 1
                break
            else:
                check = 2

        if check == 2:
            print('label', i)
            label = int(i)

            for j in os.listdir(set_floder):
                path = set_floder + "/" + j
                img = cv2.imread(path)
                face, rect = detect_face(img)

                if face is not None:
                    faces.append(face)
                    labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels

# train mo hinh bang file hinh anh da cat va xoay hinhf bang dlib
def array_data_dlib():
    faces = []
    IDs = []
    for i in os.listdir('Data_Dlib'):
        set = "../face_recogize/Data_Dlib/" + i
        label = int(i)
        for j in os.listdir(set):
            path = set + "/" + j

            faceImg = Image.open(path).convert('L')

            faceNp = np.array(faceImg)

            faces.append(faceNp)
            IDs.append(label)
            cv2.imshow("Training", faceNp)
            cv2.waitKey(1)
    return faces, IDs

# train mo hinh bang file hinh anh da cat va xoay hinhf bang haar
def array_data_haar():
    faces = []
    IDs = []
    for i in os.listdir('Data_Haar'):
        set = "../face_recogize/Data_Haar/" + i
        label = int(i)
        for j in os.listdir(set):
            path = set + "/" + j

            faceImg = Image.open(path).convert('L')

            faceNp = np.array(faceImg)

            faces.append(faceNp)
            IDs.append(label)
            cv2.imshow("Training", faceNp)
            cv2.waitKey(1)
    return faces, IDs

#Write labels to csv
def data_name(lebels):
    with open('labels_data.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(lebels)



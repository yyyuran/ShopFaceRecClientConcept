import shutil
import numpy
from PIL import Image
import os
import pickle
import cv2
import imutils
import  math
import time
import torch
from threading import Thread
import numpy as np
import tensorflow as tf
import cpbd as cp
import datetime
import datetime as dt
import Mtcnn_my
import sys
import mediapipe as mp
import argparse
import random as ran
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import SecurityUnit
import SecurityUnitServer
import sys



"""
def trace(frame, event, arg):
    print ("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace

def test():
    print ("Line 8")
    print ("Line 9")

sys.settrace(trace)
test()

"""


project_dir = os.path.dirname(os.path.abspath(__file__))
AbsPath=project_dir
def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-br', default='Br105')
    parser.add_argument('-IPcam1', default='10.57.7.34')
    parser.add_argument('-IPcam2', default='10.57.7.40')
    return parser


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    # print (namespace)

    #print("Привет, {}!".format(namespace.name))

Br=namespace.br
def initialize_model_S2048():
    # Download the pytorch model and weights.
    # Currently, it's cpu mode.
    import senet50_ft_dims_2048 as model_S2048
    network_S2048 = model_S2048.senet50_ft(weights_path=str(AbsPath)+'/'+'model/senet50_ft_dims_2048.pth')
    network_S2048.eval()
    return network_S2048


def initialize_model_R2048():
    import resnet50_ft_dims_2048 as model_S128
    network_S128 = model_S128.Resnet50_ft()
    state_dict = torch.load(str(AbsPath)+'/'+'model/resnet50_ft_dims_2048.pth')
    network_S128.load_state_dict(state_dict)
    network_S128.eval()
    return network_S128


# model_eval = initialize_model()
model_eval_S2048 = initialize_model_S2048()
# model_eval_S128 = initialize_model_S128()
model_eval_R2048 = initialize_model_R2048()


IPcam2=namespace.IPcam2


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn_SeNet1 = Mtcnn_my.MTCNN_(
    image_size=224, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
    device=device
)
from lib.face_utils import judge_side_face
from src.sort import Sort





if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    # print (namespace)

    #print("Привет, {}!".format(namespace.name))
"""
Br=namespace.br
IPcam1=namespace.IPcam1
IPcam2=namespace.IPcam2
#faceNet1 = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")


#prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
#weightsPath = os.path.sep.join([args["face"],"res10_300x300_ssd_iter_140000.caffemodel"])
#faceNet_Glasis = cv2.dnn.readNet(prototxtPath, weightsPath)

project_dir = os.path.dirname(os.path.abspath(__file__))
AbsPath=project_dir

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageProto = str(AbsPath)+'/'+"age_deploy.prototxt"
ageModel = str(AbsPath)+'/'+"age_net.caffemodel"
ageNet = cv2.dnn.readNet(ageModel, ageProto)
ageNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
ageNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
"""
"""
mtcnn_SeNet3 = Mtcnn_my.MTCNN_(
    image_size=224, margin=0, min_face_size=150,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
    device=device
)
"""

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageProto = str(AbsPath) + '/' + "age_deploy.prototxt"
ageModel = str(AbsPath) + '/' + "age_net.caffemodel"
ageNet = cv2.dnn.readNet(ageModel, ageProto)
ageNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
ageNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

genderProto =str(AbsPath)+ "/gender_deploy.prototxt"
genderModel = str(AbsPath)+"/gender_net.caffemodel"
genderList = ['Woman', 'Man']
#ageProto = "age_deploy.prototxt"
#ageModel = "age_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)
genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

conf_threshold = 0.7

# detector = dlib.get_frontal_face_detector()
# detector1 = dlib.get_frontal_face_detector()

# predictor_path = str(AbsPath)+'/'+'shape_predictor_68_face_landmarks.dat'

# predictor = dlib.shape_predictor(predictor_path)
# predictor1 = dlib.shape_predictor(predictor_path)
import torch

mean = (131.0912, 103.8827, 91.4953)
# vs = cv2.VideoCapture('rtsp://admin:12345@'+IPcam1+':554/ISAPI/Streaming/Channels/101')
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
k_W = 640 / 1920
k_h = 360 / 1080
detect_interval = 1
margin = 10
scale_rate = 0, 71
show_rate = 1.0
face_score_threshold = 0.85

project_dir = os.path.dirname(os.path.abspath(__file__))
colours = np.random.rand(32, 3)
# tracker = Sort()  # create instance of the SORT tracker

"""
# борода/
def detect_and_predict_mst(frame, faceNet, mstNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (160, 160),
                                 (104.0, 177.0, 123.0), swapRB=False)

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        preds = mstNet.predict(faces)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

Mst_Net = load_model("mustache_detector.model")
faceNet1 = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
k=0
for i, j, y in os.walk('11/' + '.'):
    print('k '+str(k))
    k=k+1
    try:
        if i[len(i) - 1:] != '.':
            for f in y:
                if (f.find('mst') == -1):
                    image = cv2.imread(i + '/' + f)
                    (locs, preds) = detect_and_predict_mst(image, faceNet1, Mst_Net)
                    for (box, pred) in zip(locs, preds):
                        # unpack the bounding box and predictions
                        (startX, startY, endX, endY) = box
                        (mask, withoutMask) = pred

                        # determine the class label and color we'll use to draw
                        # the bounding box and text
                        # label = "Mask" if mask > withoutMask else "No Mask"
                        if mask > 0.65:
                            try:
                                os.rename(i + '/' + f, i + '/' + f[:f.find("Tr")] + 'MstOn' + '_' + f[f.find("Tr"):])
                            except:
                                pass
                        else:
                            try:
                                os.rename(i + '/' + f, i + '/' + f[:f.find("Tr")] + 'MstOf' + '_' + f[f.find("Tr"):])
                            except:
                                pass
    except:
        print(" не читается файл при определении очков")

"""
"""
def improve_contrast_image_using_clahe(bgr_image: np.array) -> np.array:
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
def initialize_model_S2048():
    # Download the pytorch model and weights.
    # Currently, it's cpu mode.
    import senet50_ft_dims_2048 as model_S2048
    network_S2048 = model_S2048.senet50_ft(weights_path=str(AbsPath)+'/'+'model/senet50_ft_dims_2048.pth')
    network_S2048.eval()
    return network_S2048


def initialize_model_R2048():
    import resnet50_ft_dims_2048 as model_S128
    network_S128 = model_S128.Resnet50_ft()
    state_dict = torch.load(str(AbsPath)+'/'+'model/resnet50_ft_dims_2048.pth')
    network_S128.load_state_dict(state_dict)
    network_S128.eval()
    return network_S128

"""


# model_eval = initialize_model()
# model_eval_S2048 = initialize_model_S2048()
# model_eval_S128 = initialize_model_S128()
# model_eval_R2048 = initialize_model_R2048()

# parser.add_argument('--face_landmarks',
#                    help='Draw five face landmarks on extracted face or not ', action="store_true")


# frame = imutils.resize(vs, width=640)
def extract_eye(sh, eye_indices):
    # points = map(lambda i: sh.part(i), eye_indices)
    points = []
    for l in enumerate(eye_indices):
        points.append(sh.part(l[1]))

    return points


def extract_eye_center(sh, eye_indices):
    points = extract_eye(sh, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6


def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))


def get_rotation_matrix(p1, p2):
    angle = -angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M


def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom


def crop_image(image, det):
    left, top, right, bottom = rect_to_tuple(det)
    return image[top:bottom, left:right]


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    # return cv2.Laplacian(image, cv2.CV_64F).var()
    gray = cv2.cvtColor(imutils.resize(image, width=150), cv2.COLOR_BGR2GRAY)
    blur2 = cv2.Laplacian(gray, cv2.CV_64F).var()

    # cp.compute(gray)
    return cp.compute(gray), blur2


AnglalHorizontalLast = [0]
LatsTrackerID = 0

ListImForThisTracker = []
ListImForThisTracker_security = []
ListTrackers = []

mp_face_mesh = mp.solutions.face_mesh


# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,min_detection_confidence=0.5)


class Thread_read_cam1(Thread):
    def __init__(self):
        """Инициализация потока"""
        Thread.__init__(self)

    def run(self):
        while True:
            # Capture frame-by-frame

            try:
                pass
                # cv2.imshow('Video1', fr1)
                # cv2.waitKey(1)
            except:

                pass
            """
            try:
                cv2.imshow('Video2', SecurityUnit.fr1)
                cv2.waitKey(1)
            except:

                pass
            """
            """"
            try:
                cv2.imshow('Video2', fr2)
                cv2.waitKey(1)
            except:
                pass

            """
            time.sleep(0.05)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                # When everything is done, release the capture

        cv2.destroyAllWindows()


"""
thread_1=Thread_read_cam1()
thread_1.start()



fr1=None
fr2=None

vs = cv2.VideoCapture('rtsp://' + IPcam1 + ':554/user=admin&password=gj94hp8z&channel=1&stream=0?.sdp')






while True:
    #print('Init New ---------------------------------------------------------------------------------')
    FaceIDcounter = 0
    #gpuOpt = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, visible_device_list='0', allow_growth=False)

    #with tf.Graph().as_default():
    #    with tf.Session(config=tf.ConfigProto(gpu_options=gpuOpt,
    #                                          log_device_placement=True)) as sess:
    #pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(project_dir, "align"))
    minsize = 80  # minimum size of face for mtcnn to detect
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    directoryname = '1'
    # Glasis_Net = load_model("Glas_detector.model")
    # model = load_model("mask_detector.model")

    # mp_face_mesh = mp.solutions.face_mesh
    # face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while True:
        # Capture frame-by-frame

        # Display the resulting frame
        try:

         ret, frame_inp = vs.read()
         #frame_inp = imutils.resize(frame_inp, width=1920)
         frame = imutils.resize(frame_inp, width=640)
         if frame_inp is not None:
            # frame=improve_contrast_image_using_clahe(frame)
            if FaceIDcounter > 1000000:
                #global tracker
                tracker = None
                #tracker = Sort()
                FaceIDcounter = 0


            final_faces = []
            addtional_attribute_list = []
            # ret, frame_inp = cam.read()

            # frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)

            r_g_b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # if counter % detect_interval == 0:
            img_size = np.asarray(frame.shape)[0:2]
            # mtcnn_starttime = time()
            #faces, points = detect_face.detect_face(r_g_b_frame, minsize, pnet, rnet, onet, threshold,factor)
            #x_aligned_SeNet = mtcnn_SeNet3.detect(r_g_b_frame)

            #face_sums=len(x_aligned_SeNet)
            SaveImageForRec(face_mesh, frame, 1, FotoSize=410)

        # else:
        #    time.sleep(0.001)
        # global fr1
        # fr1= frame1
        # cv2.imshow('Video1', frame1)
        # cv2.waitKey(1)
        except:
            # vs.release()
            #vs = cv2.VideoCapture(0)
            vs = cv2.VideoCapture('rtsp://' + IPcam1 + ':554/user=admin&password=gj94hp8z&channel=1&stream=0?.sdp')
            print('Error')
            time.sleep(0.1)
        #global fr1
        try:
            fr1 = frame
        except:
            pass
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #  break
vs.release()

"""


#######


class Tr_DeleteFolder(Thread):
    def __init__(self):
        """Инициализация потока"""
        Thread.__init__(self)

    def run(self):
        while True:
            for i, j, y in os.walk(str(AbsPath) + '/' + 'FaceRecogition/.'):
                if i != 'FaceRecogition/.':
                    if len(y) == 0:
                        try:
                            os.rmdir(i)
                            # print('Folder deleted: '+ str(i))
                        except:
                            pass
            for i, j, y in os.walk(str(AbsPath) + '/' + 'FaceRecogition_adv_1/.'):
                if i != 'FaceRecogition/.':
                    if len(y) == 0:
                        try:
                            os.rmdir(i)
                            # print('Folder deleted: '+ str(i))
                        except:
                            pass
            for i, j, y in os.walk(str(AbsPath) + '/' + 'FaceRecogition_Security/.'):
                if i != 'FaceRecogition_Security/.':
                    if len(y) == 0:
                        try:
                            os.rmdir(i)
                            # Vtprint('Folder in FaceRecogition_Security deleted: ' + str(i))
                        except:
                            pass

            time.sleep(0.3)


def detect_and_predict_Glas(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (160, 160),
                                 (104.0, 177.0, 123.0), swapRB=False)

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        preds = maskNet.predict(faces)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


def RenameFoders(model, Glasis_Net, Directory, faceNet1):
    for i, j, y in os.walk(Directory + '.'):
        # k = 0
        if i[len(i) - 1:] != '.':
            try:
                # d = (int(i[i.index('/') + 1:][i[i.index('/') + 1:].index('/') + 1:]))
                # if len(y) == 0:
                #    os.rmdir(i)
                if len(i[i.index('.') + 2:]) < 6:
                    for f in y:
                        try:
                            if f.find('adv') > -1:
                                dTime = f[f.index('_') + 1:][f[f.index('_') + 1:].index('_') + 1:][
                                        :f[f.index('_') + 1:][f[f.index('_') + 1:].index('_') + 1:].index('adv') - 1]
                            else:
                                dTime = f[f.index('_') + 1:][f[f.index('_') + 1:].index('_') + 1:][
                                        :f[f.index('_') + 1:][f[f.index('_') + 1:].index('_') + 1:].index('Tr') - 1]

                            os.rename(i, Directory + dTime)

                        except:
                            pass
                        break
            except:
                pass
    # удалим файлы из паок более 15
    for i, j, y in os.walk(Directory + '.'):
        # k = 0
        if i[len(i) - 1:] != '.':
            # if len(y) == 0:
            #    os.rmdir(i)
            if (len(y) > 80):
                ListFilesForDel = []
                ListNamesFilesForDel = []
                CountFilesForDel = len(y) - 80
                while (len(ListFilesForDel)) < (CountFilesForDel):
                    indFileForDel = int(ran.random() * len(y))
                    try:
                        if (ListFilesForDel.index(indFileForDel)):
                            pass
                    except:
                        ListFilesForDel.append(indFileForDel)
                        ListNamesFilesForDel.append(y[indFileForDel])

                for f in ListNamesFilesForDel:
                    try:
                        os.remove(i + '/' + f)
                    except:
                        pass

    ##проверим налдиче  очков

    for i, j, y in os.walk(Directory + '.'):
        try:
            if i[len(i) - 1:] != '.':
                for f in y:
                    if (f.find('Gl') == -1):
                        image = cv2.imread(i + '/' + f)
                        (locs, preds) = detect_and_predict_Glas(image, faceNet1, Glasis_Net)
                        for (box, pred) in zip(locs, preds):
                            # unpack the bounding box and predictions
                            (startX, startY, endX, endY) = box
                            (mask, withoutMask) = pred

                            # determine the class label and color we'll use to draw
                            # the bounding box and text
                            # label = "Mask" if mask > withoutMask else "No Mask"
                            if mask > 0.50:
                                try:
                                    os.rename(i + '/' + f, i + '/' + f[:f.find("Tr")] + 'GlOn' + '_' + f[f.find("Tr"):])
                                except:
                                    pass
                            else:
                                try:
                                    os.rename(i + '/' + f, i + '/' + f[:f.find("Tr")] + 'GlOf' + '_' + f[f.find("Tr"):])
                                except:
                                    pass
        except:
            print(" не читается файл при определении очков")
    # проверим налдиче маски
    # print("Dane")

        # gender
    for i, j, y in os.walk(Directory + '.'):
        # k = 0
        try:
            if i[len(i) - 1:] != '.':
                if ((i[len(i) - 2:] != 'Of') & (i[len(i) - 2:] != 'On')):
                    for f in y:
                        newFileName = (i + '/' + f)[:len(i + '/' + f) - 4]
                        image = cv2.imread(i + '/' + f)
                        blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), MODEL_MEAN_VALUES, swapRB=False)
                        genderNet.setInput(blob)
                        genderPreds = genderNet.forward()
                        gender = genderList[genderPreds[0].argmax()]
                        if f.find('an')<0:
                            os.rename(i + '/' + f, newFileName + '_' + gender + '.bmp')

        except:
            print(" проблема при определении пола")
    Age1=0
    for i, j, y in os.walk(Directory + '.'):
        # k = 0
        try:
            if i[len(i) - 1:] != '.':
                # if len(y) == 0:
                #    os.rmdir(i)
                if ((i[len(i) - 2:] != 'Of') & (i[len(i) - 2:] != 'On')):

                    MassivAgeForCurrentFolder = []
                    for f in y:
                        image = cv2.imread(i + '/' + f)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (224, 224))
                        blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), MODEL_MEAN_VALUES, swapRB=False)
                        ageNet.setInput(blob)
                        agePreds = ageNet.forward()
                        m = max(agePreds[0])
                        Age1 = int((np.where((agePreds[0]) == m))[0])
                        MassivAgeForCurrentFolder.append(Age1)

                    # удалим мин и макс года - шумят
                    MassTemp = []
                    for elem in MassivAgeForCurrentFolder:
                        if (elem not in MassTemp):
                            MassTemp.append(elem)
                    if len(MassTemp) >= 3:
                        minAge = min(MassTemp)
                        maxAge = max(MassTemp)
                        NewMassivAge = []
                        for elem in MassivAgeForCurrentFolder:
                            if (elem != minAge) & (elem != maxAge):

                                NewMassivAge.append(elem)
                        Age1 = int(sum(NewMassivAge) / len(NewMassivAge))
                    else:
                        Age1 = int(sum(MassivAgeForCurrentFolder) / len(MassivAgeForCurrentFolder))
                    for f in y:
                        try:
                            newFileName = (i + '/' + f)[:len(i + '/' + f) - 4]
                            if len(str(Age1))==1:
                                temp_str='0'+str(Age1)
                            else:
                                temp_str=str(Age1)
                            try:
                                int(f[(len(f) - 6):(len(f) - 4)])
                            except:
                                os.rename(i + '/' + f, newFileName + '_' + temp_str + '.bmp')
                        except:
                            Age1 = 0
                            pass
        except:
            Age1 = 0
            print(" не читается файл при определении Age")
    if Age1>0:
        for i, j, y in os.walk(Directory + '.'):
            # k = 0
            try:
                if i[len(i) - 1:] != '.':
                    findedAge=False
                    for f in y:
                        if int(f[(len(f)-6):(len(f)-4)])>0:
                            findedAge = True
                        break

                    # if len(y) == 0:
                    #    os.rmdir(i)
                    if findedAge == True:
                        if ((i[len(i) - 2:] != 'Of') & (i[len(i) - 2:] != 'On')):
                            maskOn = 0
                            maskOff = 0
                            for f in y:

                                image = cv2.imread(i + '/' + f)

                                # (locs, preds) = detect_and_predict_Glas(image, faceNet1, Glasis_Net)
                                ############
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                image = cv2.resize(image, (224, 224))
                                """
                                blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), MODEL_MEAN_VALUES, swapRB=False)
                                ageNet.setInput(blob)
                                agePreds = ageNet.forward()
                                m = max(agePreds[0])
                                Age1 = int((np.where((agePreds[0]) == m))[0])
                                try:
                                    newFileName = (i + '/' + f)[:len(i + '/' + f) - 4]
                                    os.rename(i + '/' + f, newFileName + '_' + str(Age1) + '.bmp')
                                except:
                                    pass
                                """
                                image = img_to_array(image)
                                image = preprocess_input(image)
                                image = np.expand_dims(image, axis=0)
                                (mask, withoutMask) = model.predict(image)[0]

                                if mask > 0.80:
                                    maskOn = maskOn + 1
                                else:
                                    maskOff = maskOff + 1

                            if maskOn > maskOff:
                                try:
                                    os.rename(i, i + '_MaskOn')

                                    if (Directory.find('Secur') == -1):
                                        src = i + '_MaskOn'
                                        dest = str(AbsPath) + '/' + 'DISTR/FaceRecogition_old/' + i[
                                                                                                  i.index('/') + 3:] + '_MaskOn'
                                        # shutil.copytree(src, dest)
                                except:
                                    pass
                            else:
                                try:
                                    os.rename(i, i + '_MaskOf')
                                    if (Directory.find('Secur') == -1):
                                        src = i + '_MaskOf'
                                        dest = str(AbsPath) + '/' + 'DISTR/FaceRecogition_old/' + i[
                                                                                                  i.index('/') + 3:] + '_MaskOf'
                                        # shutil.copytree(src, dest)
                                except:
                                    pass
            except:
                print(" не читается файл при определении маски")





    # сдесь временно возраст определим
    """
    for i, j, y in os.walk('FaceRecogition/.'):
        # k = 0
        if i[len(i) - 1:] != '.':
            # if len(y) == 0:
            #    os.rmdir(i)

            Ages = []

            for f in y:

                image=cv2.imread(i+'/'+f)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                #image = img_to_array(image)
                #image = preprocess_input(image)
                #image = np.expand_dims(image, axis=0)
                #(mask, withoutMask) = model.predict(image)[0]

                blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224),
                                             MODEL_MEAN_VALUES, swapRB=False)

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                # age=ageList[agePreds[0].argmax()]
                m = max(agePreds[0])
                Age1= int((np.where((agePreds[0]) == m))[0])
                # print(f'Age: {Age} years')
                # FileName=FileName+'_'+gender+'_'+str(Age)+'_'+str(int(fm))
                #FileName = FileName + '_' + gender + '_' + str(Age)
                Ages.append(Age1)
                try:
                    newFileName=(i+'/'+f)[:len(i+'/'+f)-4]
                    os.rename(i+'/'+f, newFileName+'_'+str(Age1)+'.bmp')
                except:
                    pass

            FolderAge1=int(sum(Ages)/len(Ages))


            try:
                # os.rename(i, i+'_'+str(int(sum(Ages)/len(Ages))))
                #os.rename(i, i[:len(i) - 3])

                pass
            except:
                pass
    """


class tread_FaceRecProc(Thread):
    def __init__(self):
        """Инициализация потока"""
        Thread.__init__(self)



    def run(self):
        model = load_model(str(AbsPath) + '/' + "mask_detector.model")
        Glasis_Net = load_model(str(AbsPath) + '/' + "Glas_detector.model")
        faceNet1 = cv2.dnn.readNet(str(AbsPath) + '/' + "deploy.prototxt",
                                   str(AbsPath) + '/' + "res10_300x300_ssd_iter_140000.caffemodel")
        Conter_S2048 = 0
        Conter_R2048 = 0

        def improve_contrast_image_using_clahe(image):
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(1, 1))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return image

        def AppendFileInFolder(name, fn, enc_R2048, enc_S2048):
            try:
                list = os.listdir(
                    'FoldersForImages/' + str(name))  # dir is your directory path
            except Exception as e:
                list = []
            number_files = len(list)
            if number_files <= 90:
                # data_SeNet['encodings'].append(encoding_SeNet)
                # data_resnet['encodings'].append(embeddin)
                data_S2048['encodings'].append(enc_S2048)
                data_R2048['encodings'].append(enc_R2048)
                # data_S128['encodings'].append(encoding_S128)
                # data_facerec['encodings'].append(encoding)

                # data_SeNet['names'].append(name)
                data_S2048['names'].append(name)
                data_R2048['names'].append(name)

                data_ListFilesNames['ListFilesNames'].append(fn)


            else:
                # удаляем самый старый файл и ключи из память
                # адо определить все индексы файлов для этой папки
                ss = [a for a, b in enumerate(data_S2048["names"]) if b == name]  # сриок индексов этой папке
                minindexInFolder = min(ss)
                del (data_S2048['names'][minindexInFolder])
                del (data_R2048['names'][minindexInFolder])

                del (data_R2048['encodings'][minindexInFolder])
                del (data_S2048['encodings'][minindexInFolder])
                try:
                    os.rename(
                        str(AbsPath) + '/' + 'FoldersForImages/' + name + '/' + data_ListFilesNames['ListFilesNames'][
                            minindexInFolder],
                        str(AbsPath) + '/' + 'FoldersForImages/' + name + '/' + '_' +
                        data_ListFilesNames['ListFilesNames'][
                            minindexInFolder])
                except:
                    pass
                del (data_ListFilesNames['ListFilesNames'][minindexInFolder])

                # новый долбавляем
                data_S2048['encodings'].append(enc_S2048)
                data_R2048['encodings'].append(enc_R2048)
                data_S2048['names'].append(name)
                data_R2048['names'].append(name)
                data_ListFilesNames['ListFilesNames'].append(fn)

        def NewPhotoId():
            data2 = pickle.loads(open(str(AbsPath) + '/' + "id_face_last.pickle", "rb").read())
            id_last = data2['id_face_last'] + 1
            data2 = {"id_face_last": (id_last)}
            f = open(str(AbsPath) + '/' + 'id_face_last.pickle', "wb")
            f.write(pickle.dumps(data2))
            f.close()

            from pathlib import Path

            Path(str(FolderOutput + str(id_last))).mkdir(parents=True, exist_ok=True)
            os.chmod(FolderOutput + str(id_last), 0o777)
            for NameAndimage_forNewFolder in ListImagesForCurrentFolder:
                AppendFileInFolder(str(id_last), NameAndimage_forNewFolder[0], NameAndimage_forNewFolder[2],
                                   NameAndimage_forNewFolder[3])

                # if os.path.exists(str('FoldersForImages/' + str(id_last)) + '/' + NameAndimage_forNewFolder[0]) == True:
                #    print('********************************************************************************************************************')

                if (cv2.imwrite(str(FolderOutput + str(id_last)) + '/' + NameAndimage_forNewFolder[0],
                                NameAndimage_forNewFolder[1])) == False:
                    print(
                        '*****************************************************error*save***************************************************************')
                else:
                    os.chmod(str(FolderOutput + str(id_last)) + '/' + NameAndimage_forNewFolder[0], 0o777)

        def detect_and_predict_mask(frame, faceNet, maskNet, name):
            # grab the dimensions of the frame and then construct a blob
            # from it
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                         (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the face detections
            faceNet.setInput(blob)
            detections = faceNet.forward()

            # initialize our list of faces, their corresponding locations,
            # and the list of predictions from our face mask network
            faces = []
            locs = []
            preds = []

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the detection
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # ensure the bounding boxes fall within the dimensions of
                    # the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    # extract the face ROI, convert it from BGR to RGB channel
                    # ordering, resize it to 224x224, and preprocess it
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    face = np.expand_dims(face, axis=0)

                    # add the face and bounding boxes to their respective
                    # lists
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))

            # only make a predictions if at least one face was detected
            if len(faces) > 0:
                # for faster inference we'll make batch predictions on *all*
                # faces at the same time rather than one-by-one predictions
                # in the above `for` loop
                preds = maskNet.predict(faces)

            # return a 2-tuple of the face locations and their corresponding
            # locations
            return (locs, preds)

        ListFodersForRec = []
        FoldersFullRec = []
        FoldersRecAdv1 = []
        FoldersFullRec.append(str(AbsPath) + '/' + 'DISTR/FaceRecogition_temp/')
        FoldersFullRec.append(str(AbsPath) + '/' + 'FaceRecogition/')

        FoldersRecAdv1.append(str(AbsPath) + '/' + 'DISTR/FaceRecogition_temp_adv_1/')
        FoldersRecAdv1.append(str(AbsPath) + '/' + 'FaceRecogition_adv_1/')

        ListFodersForRec.append(FoldersFullRec)
        ListFodersForRec.append(FoldersRecAdv1)
        while True:
            for ff in ListFodersForRec:
                try:
                    FolderInput = ff[0]
                    FolderOutput = ff[1]
                    data_S2048 = pickle.loads(open(str(AbsPath) + '/' + "2009_S2048.pickle", "rb").read())
                    data_R2048 = pickle.loads(open(str(AbsPath) + '/' + "2009_R2048.pickle", "rb").read())
                    data_ListFilesNames = pickle.loads(open(str(AbsPath) + '/' + "ListFilesNames.pickle", "rb").read())

                    f = open(str(AbsPath) + '/' + '2009_R2048.pickle', "wb")
                    data_R2048.clear()
                    f.write(pickle.dumps(data_R2048))
                    f.close()

                    f = open(str(AbsPath) + '/' + '2009_S2048.pickle', "wb")
                    data_S2048.clear()
                    f.write(pickle.dumps(data_S2048))
                    f.close()

                    f = open(str(AbsPath) + '/' + 'ListFilesNames.pickle', "wb")
                    data_ListFilesNames.clear()
                    f.write(pickle.dumps(data_ListFilesNames))
                    f.close()

                    knownEncodings_S2048 = []
                    knownNames_S2048 = []

                    knownEncodings_R2048 = []
                    knownNames_R2048 = []

                    FilseNames = []

                    data_R2048 = {"encodings": knownEncodings_R2048, "names": knownNames_R2048}
                    f = open(str(AbsPath) + '/' + '2009_R2048.pickle', "wb")
                    f.write(pickle.dumps(data_R2048))
                    f.close()

                    data_S2048 = {"encodings": knownEncodings_S2048, "names": knownNames_S2048}
                    f = open(str(AbsPath) + '/' + '2009_S2048.pickle', "wb")
                    f.write(pickle.dumps(data_S2048))
                    f.close()

                    data_ListFilesNames = {"ListFilesNames": FilseNames}
                    f = open(str(AbsPath) + '/' + 'ListFilesNames.pickle', "wb")
                    f.write(pickle.dumps(data_ListFilesNames))
                    f.close()

                    id_last = 0
                    data2 = {"id_face_last": (id_last)}
                    f = open(str(AbsPath) + '/' + 'id_face_last.pickle', "wb")
                    f.write(pickle.dumps(data2))
                    f.close()

                    for FullDirName, j, filesInCurrentFolder in os.walk(FolderInput + '.'):
                        try:
                            dirName = FullDirName[len(FullDirName) - 1:]
                            if (dirName != '.') & ((dirName != '..')):
                                FileFlag = ''
                                ListImagesForCurrentFolder = []
                                # filesInCurrentFolder = conn.listPath('DISTR', 'FaceRecogition/' + file.filename)
                                CountPositiveRecognized = 0
                                ListRecognizedNamesFolders = []
                                for fileInFolder in filesInCurrentFolder:
                                    if len(ListImagesForCurrentFolder) == 90:
                                        break
                                    if (fileInFolder != '.') & ((fileInFolder != '..')):
                                        try:
                                            from pathlib import Path

                                            FileName = fileInFolder[:len(fileInFolder) - 4]
                                            # file_obj = tempfile.NamedTemporaryFile()
                                            # file_attributes, filesize = conn.retrieveFile('DISTR',
                                            #                                              'FaceRecogition/' + file.filename + '/' + fileInFolder.filename,
                                            #                                              file_obj)
                                            # conn.deleteFiles('FaceRecogition/' + file.filename, fileInFolder.filename)
                                            pil_image = Image.open(open(FullDirName + '/' + fileInFolder, 'rb'))
                                            image = cv2.cvtColor(((numpy.array(pil_image))), cv2.COLOR_RGB2BGR)
                                            if (image is None):
                                                print(
                                                    '------------------------------Пустой файл --------------------------')
                                            # Track = file.filename
                                            # EncodingsForCurrentTrack=[]

                                            if ((image is not None)):

                                                aligned = []
                                                aligned_SeNet = []

                                                x_aligned_SeNet, prob_SeNet = mtcnn_SeNet1(improve_contrast_image_using_clahe(image), return_prob=True)
                                                if prob_SeNet is not None:
                                                    if ((prob_SeNet < 0.5)):
                                                        print(
                                                            "**********************************************************************************************************************************************")
                                                    if ((prob_SeNet > 0.5)):
                                                        # c = c + 1
                                                        # print(str(c))
                                                        Imgage_NDArray = cv2.cvtColor(x_aligned_SeNet, cv2.COLOR_RGB2BGR)
                                                        Imgage_NDArray = Imgage_NDArray - mean
                                                        temparr = np.ndarray(shape=(1, 224, 224, 3))
                                                        temparr[0] = Imgage_NDArray
                                                        # face_feats = np.empty((1, 256))
                                                        ten = torch.Tensor(temparr.transpose(0, 3, 1, 2))
                                                        ten_dev = ten.to(device)

                                                        face_feats_S2048 = np.empty((1, 2048))
                                                        # ten = torch.Tensor(temparr.transpose(0, 3, 1, 2))
                                                        # ten_dev = ten.to(device)
                                                        f_S2048 = model_eval_S2048(ten_dev)[1].detach().cpu().numpy()[:, :,
                                                                  0, 0]
                                                        face_feats_S2048[0:1] = f_S2048 / np.sqrt(
                                                            np.sum(f_S2048 ** 2, -1, keepdims=True))
                                                        encoding_S2048 = face_feats_S2048
                                                        #####################################################################################################
                                                        face_feats_R2048 = np.empty((1, 2048))
                                                        f_R2048 = model_eval_R2048(ten_dev)[1].detach().cpu().numpy()[:, :,
                                                                  0, 0]
                                                        face_feats_R2048[0:1] = f_R2048 / np.sqrt(
                                                            np.sum(f_R2048 ** 2, -1, keepdims=True))
                                                        encoding_R2048 = face_feats_R2048

                                                        FileName = FileName + '.bmp'

                                                        ListNameFilesAndFiles = []
                                                        ListNameFilesAndFiles.append(FileName)
                                                        ListNameFilesAndFiles.append(image)
                                                        ListNameFilesAndFiles.append(encoding_R2048)
                                                        ListNameFilesAndFiles.append(encoding_S2048)
                                                        ListImagesForCurrentFolder.append(ListNameFilesAndFiles)
                                                        # begin
                                                        names = []
                                                        # for encoding in encodings:
                                                        name = "Unknown"
                                                        if (len(data_S2048['encodings']) != 0) & (FileFlag != 'First'):

                                                            dists_S2048 = [[1 - np.dot(encoding_S2048, e2.T)[0][0] for e2 in
                                                                            data_S2048['encodings']]][0]
                                                            dists_R2048 = [[1 - np.dot(encoding_R2048, e2.T)[0][0] for e2 in
                                                                            data_R2048['encodings']]][0]

                                                            distanceArrayList_S2048 = [b for (i, b) in
                                                                                       enumerate(dists_S2048)]
                                                            distanceArrayList_R2048 = [b for (i, b) in
                                                                                       enumerate(dists_R2048)]
                                                            # distanceArrayList_S128 = [b for (i, b) in
                                                            #                          enumerate(dists_S128)]

                                                            matches_S2048 = []
                                                            for d in enumerate(dists_S2048):
                                                                # if (d[1]<0.735):
                                                                # if (d[1] < 0.276):
                                                                if (d[1] < 0.286):
                                                                    matches_S2048.append(True)
                                                                else:
                                                                    matches_S2048.append(False)

                                                            matches_R2048 = []
                                                            for d in enumerate(dists_R2048):
                                                                # if (d[1]<0.735):
                                                                # if (d[1] < 0.316):
                                                                if (d[1] < 0.326):
                                                                    matches_R2048.append(True)
                                                                else:
                                                                    matches_R2048.append(False)
                                                            # matches_S128 = []

                                                            matchedIdxs_S2048 = [i for (i, b) in enumerate(matches_S2048) if
                                                                                 b]
                                                            matchedIdxs_R2048 = [i for (i, b) in enumerate(matches_R2048) if
                                                                                 b]
                                                            # matchedIdxs_S128 = [i for (i, b) in enumerate(matches_S128) if
                                                            #                    b]

                                                            # matchedIdxs_facerec = [i for (i, b) in enumerate(matches_facerec) if b]

                                                            # Conter_SeNet = Conter_SeNet + len(matchedIdxs_SeNet)
                                                            Conter_S2048 = Conter_S2048 + len(matchedIdxs_S2048)
                                                            Conter_R2048 = Conter_R2048 + len(matchedIdxs_R2048)

                                                            print(' Counter_SeNet: '
                                                                  # +str(Conter_SeNet) + ' Counter_S2048: '
                                                                  + str(Conter_S2048) + ' Counter_R2048: '
                                                                  + str(Conter_R2048) + ' Counter_S128: ')
                                                            # + str(Conter_S128))

                                                            # устраняем те труе котьорые не общие для всех
                                                            for x in range(0, len(matches_R2048)):
                                                                if ((matches_S2048[x] != True) or (
                                                                        matches_R2048[x] != True)):
                                                                    # matches_SeNet[x] = False
                                                                    matches_S2048[x] = False
                                                                    matches_R2048[x] = False
                                                                    # matches_S128[x] = False

                                                            # ListFolderswidthTrue = [b for a, b in
                                                            #                        enumerate(data_S2048["names"]) if a in (
                                                            #                        [a for a, b in enumerate(matches_S2048)
                                                            #                        if b])]  # список паок где есть труе

                                                            ListFolderswidthTrue = []  # список паок где есть труе
                                                            ListTrueIndexes = (
                                                                [a for a, b in enumerate(matches_S2048) if b])
                                                            for ind in ListTrueIndexes:
                                                                ListFolderswidthTrue.append(data_S2048["names"][ind])

                                                            ListFolderswidthTrueGroup = []  # сгруппировал папки
                                                            for elem in ListFolderswidthTrue:
                                                                if elem not in ListFolderswidthTrueGroup:
                                                                    ListFolderswidthTrueGroup.append(elem)

                                                            # ложно положительные уменьшаем с помощть проверки на количество фоток с труе в папке
                                                            for elem in ListFolderswidthTrueGroup:
                                                                ss = [a for a, b in enumerate(data_S2048["names"]) if
                                                                      b == elem]  # сриок индексов этой папке
                                                                CountElemInFolder = len(ss)
                                                                countTrueInFolder = len(
                                                                    [b for a, b in enumerate(matches_S2048) if
                                                                     (a in ss) & (b)])
                                                                # if ((CountElemInFolder >= 3) & (CountElemInFolder < 6) & (countTrueInFolder < 2)) or (((CountElemInFolder >= 6) & (CountElemInFolder < 10) & (countTrueInFolder < 3))) or (((CountElemInFolder >= 10) & (CountElemInFolder < 2000) & (countTrueInFolder < 4))):
                                                                if (((CountElemInFolder >= 3) & (CountElemInFolder < 5) & (
                                                                        countTrueInFolder < 2)) or (
                                                                        (CountElemInFolder >= 5) & (
                                                                        countTrueInFolder < int(CountElemInFolder / 3.5)))):
                                                                    for ind in ss:
                                                                        # matches_SeNet[ind] = False
                                                                        matches_S2048[ind] = False
                                                                        matches_R2048[ind] = False
                                                                        # matches_S128[ind] = False

                                                            if (True in matches_S2048) or (
                                                                    True in matches_R2048):  ###почему или!!!!!!!!!!!!!!!!!!!!!!!!!!!ане и
                                                                # покупатель распознан
                                                                # matchedIdxs_resnet = [i for (i, b) in enumerate(matches_resnet) if b]

                                                                matchedIdxs_S2048 = [i for (i, b) in
                                                                                     enumerate(matches_S2048) if b]

                                                                matchedIdxs_R2048 = [i for (i, b) in
                                                                                     enumerate(matches_R2048) if b]
                                                                counts = {}
                                                                agePerFolder_R2048 = {}
                                                                for i in matchedIdxs_R2048:
                                                                    name_R2048 = data_R2048["names"][i]
                                                                    counts[name_R2048] = counts.get(name_R2048, 0) + 1
                                                                    agePerFolder_R2048[
                                                                        name_R2048] = agePerFolder_R2048.get(name_R2048,
                                                                                                             0) + \
                                                                                      distanceArrayList_R2048[i]

                                                                for i in counts:
                                                                    agePerFolder_R2048[i] = agePerFolder_R2048[
                                                                                                i] / counts.get(i, 0)
                                                                    # name = data["names"][i]
                                                                #####################
                                                                counts = {}
                                                                agePerFolder_S2048 = {}
                                                                for i in matchedIdxs_S2048:
                                                                    name_S2048 = data_S2048["names"][i]
                                                                    counts[name_S2048] = counts.get(name_S2048, 0) + 1
                                                                    agePerFolder_S2048[
                                                                        name_S2048] = agePerFolder_S2048.get(name_S2048,
                                                                                                             0) + \
                                                                                      distanceArrayList_S2048[i]

                                                                for i in counts:
                                                                    agePerFolder_S2048[i] = agePerFolder_S2048[
                                                                                                i] / counts.get(i, 0)
                                                                    # name = data["names"][i]

                                                                # if  (len(matchedIdxs_SeNet)>1)or(len(matchedIdxs_resnet)>1)or(len(matchedIdxs_facerec)>1):
                                                                #    pass
                                                                # SeNetFodersObsie = {}
                                                                S2048FodersObsie = {}
                                                                R2048FodersObsie = {}
                                                                # S128FodersObsie = {}

                                                                # facerecFodersObsie = {}
                                                                for r in [agePerFolder_S2048][0]:
                                                                    for f in [agePerFolder_R2048][0]:
                                                                        # for s in [agePerFolder_SeNet][0]:
                                                                        # for s128 in [agePerFolder_S128][0]:
                                                                        if (f == r):
                                                                            dst_r = [agePerFolder_S2048][0][r]
                                                                            dst_f = [agePerFolder_R2048][0][f]
                                                                            # dst_s = [agePerFolder_SeNet][0][s]
                                                                            # dst_s128 = [agePerFolder_S128][0][s128]
                                                                            S2048FodersObsie[r] = dst_r
                                                                            R2048FodersObsie[f] = dst_f
                                                                            # SeNetFodersObsie[s] = dst_s
                                                                            # S128FodersObsie[s128] = dst_s128

                                                                # name = max(counts, key=counts.get)
                                                                ##IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII

                                                                if len(S2048FodersObsie) > 0:  #######это временно устанвил

                                                                    S2048MinDist = S2048FodersObsie[
                                                                        min(S2048FodersObsie, key=agePerFolder_S2048.get)]
                                                                    S2048MinDistNmae = min(S2048FodersObsie,
                                                                                           key=agePerFolder_S2048.get)

                                                                    R2048MinDist = R2048FodersObsie[
                                                                        min(R2048FodersObsie, key=agePerFolder_R2048.get)]
                                                                    R2048MinDistNmae = min(R2048FodersObsie,
                                                                                           key=agePerFolder_R2048.get)

                                                                    S2048_dist_correct = S2048MinDist * (0.306 / 0.266)

                                                                    # S128_dist_correct = S128MinDist * (0.324 / 2.3)

                                                                    # name=R2048MinDistNmae

                                                                    if (R2048MinDist < S2048_dist_correct):
                                                                        name = R2048MinDistNmae
                                                                        # Conter_facenet = Conter_facenet+1

                                                                    if (S2048_dist_correct < R2048MinDist):
                                                                        name = S2048MinDistNmae
                                                                        # Conter_resnet = Conter_resnet + 1

                                                                    # knownEncodings.append(encoding)
                                                                    # knownNames.append(name)
                                                                    FileFlag = 'Recognized'
                                                                    ListRecognizedNamesFolders.append(name)


                                                                else:
                                                                    # новый
                                                                    FileFlag = 'New'
                                                                    # NewPhotoId()



                                                            else:
                                                                # новый
                                                                # NewPhotoId()
                                                                FileFlag = 'New'

                                                        else:  # cv2.imshow('image',imag

                                                            FileFlag = 'First'

                                                            # первый покупатель
                                                            """
    
    
                                                        """


                                            else:
                                                print(
                                                    '--------------------------------------------------------------------------------')
                                        except:
                                            print(
                                                '-------------------------------------------------------------error-------------------')

                                for fileInFolder in filesInCurrentFolder:
                                    if (fileInFolder != '.') & (fileInFolder != '..'):
                                        try:
                                            # conn.deleteFiles('FaceRecogition', file.filename + '/' + fileInFolder.filename)
                                            os.remove(FullDirName + '/' + fileInFolder)
                                        except:
                                            pass

                                try:
                                    # conn.deleteFiles('FaceRecogition/', file.filename)
                                    try:
                                        os.rmdir(FullDirName)
                                        print('Folder deleted: ' + str(i))
                                    except:
                                        pass
                                except:
                                    pass

                                if FileFlag != 'First':
                                    ListRecognizedNamesFoldersGrouped_SumPositiveRec = []
                                    ListRecognizedNamesFoldersGrouped = []  # сгруппировал папки
                                    for elem in ListRecognizedNamesFolders:
                                        if elem not in ListRecognizedNamesFoldersGrouped:
                                            ListRecognizedNamesFoldersGrouped.append(elem)
                                            ListRecognizedNamesFoldersGrouped_SumPositiveRec.append(0)
                                    ind = 0
                                    # тут  лажа
                                    for elem in ListRecognizedNamesFoldersGrouped:
                                        for elem1 in ListRecognizedNamesFolders:
                                            if elem == elem1:
                                                ListRecognizedNamesFoldersGrouped_SumPositiveRec[ind] = \
                                                    ListRecognizedNamesFoldersGrouped_SumPositiveRec[ind] + 1
                                        ind = ind + 1
                                    if len(ListRecognizedNamesFoldersGrouped_SumPositiveRec) > 0:
                                        CountPositiveRecognized = max(ListRecognizedNamesFoldersGrouped_SumPositiveRec)
                                        # if (CountPositiveRecognized > int(len(ListImagesForCurrentFolder[0]) / 3)):# тут ошибкаlen(ListImagesForCurrentFolder)
                                        # if (CountPositiveRecognized > int(len(ListImagesForCurrentFolder) / 5)):  # тут ошибкаlen(ListImagesForCurrentFolder)
                                        if ((
                                                ((len(ListImagesForCurrentFolder) >= 3) & (
                                                        len(ListImagesForCurrentFolder) < 5) & (
                                                         CountPositiveRecognized < 2))) or (
                                                (CountPositiveRecognized < int(
                                                    len(ListImagesForCurrentFolder) / 3.5)) & (
                                                        (len(ListImagesForCurrentFolder) >= 5)))):
                                            # or (((len(ListImagesForCurrentFolder) >= 6) & (len(ListImagesForCurrentFolder) < 10) & (CountPositiveRecognized < 3))) or (((len(ListImagesForCurrentFolder) >= 10) & (len(ListImagesForCurrentFolder) < 2000) & (CountPositiveRecognized < 4))):
                                            NewPhotoId()
                                        else:

                                            name = ListRecognizedNamesFoldersGrouped[
                                                ListRecognizedNamesFoldersGrouped_SumPositiveRec.index(
                                                    max(ListRecognizedNamesFoldersGrouped_SumPositiveRec))]  # добавил

                                            # распознал
                                            for NameAndimage_forNewFolder in ListImagesForCurrentFolder:
                                                AppendFileInFolder(str(name), NameAndimage_forNewFolder[0],
                                                                   NameAndimage_forNewFolder[2],
                                                                   NameAndimage_forNewFolder[3])
                                                # fl = str('FoldersForImages/' + str(name)) + '/' + NameAndimage_forNewFolder[0]
                                                # if os.path.exists(fl) ==True:
                                                #    print('********************************************************************************************************************')

                                                if (cv2.imwrite(
                                                        str(FolderOutput + str(name)) + '/' + NameAndimage_forNewFolder[
                                                            0], NameAndimage_forNewFolder[1])) == False:
                                                    print(
                                                        '*****************************************************error*save***************************************************************')
                                                else:
                                                    os.chmod(
                                                        str(FolderOutput + str(name)) + '/' + NameAndimage_forNewFolder[
                                                            0], 0o777)


                                    else:
                                        NewPhotoId()

                                else:
                                    for NameAndimage_forNewFolder in ListImagesForCurrentFolder:
                                        folder = FileName[:FileName.index('_')]
                                        # ListFoders_origin.append(0)
                                        # ListFoders.append(folder)
                                        # if Track != None:
                                        #    if Track not in Tracks:
                                        #        Tracks.append(Track)
                                        #        Track_Folder[Track] = 0

                                        data_S2048['names'].append('0')
                                        data_R2048['names'].append('0')
                                        data_S2048['encodings'].append(NameAndimage_forNewFolder[3])
                                        data_R2048['encodings'].append(NameAndimage_forNewFolder[2])

                                        data_ListFilesNames['ListFilesNames'].append(NameAndimage_forNewFolder[0])

                                        # data_S2048 = {"encodings": knownEncodings_S2048,
                                        #              "names": knownNames_S2048}
                                        f = open(str(AbsPath) + '/' + '2009_S2048.pickle', "wb")
                                        f.write(pickle.dumps(data_S2048))
                                        f.close()

                                        # data_R2048 = {"encodings": knownEncodings_R2048,
                                        #              "names": knownNames_R2048}
                                        f = open(str(AbsPath) + '/' + '2009_R2048.pickle', "wb")
                                        f.write(pickle.dumps(data_R2048))
                                        f.close()

                                        data2 = {"id_face_last": 0}
                                        f = open(str(AbsPath) + '/' + 'id_face_last.pickle', "wb")
                                        f.write(pickle.dumps(data2))
                                        f.close()

                                        from pathlib import Path

                                        Path(FolderOutput + '0').mkdir(parents=True, exist_ok=True)
                                        os.chmod(FolderOutput + '0', 0o777)
                                        now = dt.datetime.now()
                                        cv2.imwrite(FolderOutput + '0/' + NameAndimage_forNewFolder[0],
                                                    NameAndimage_forNewFolder[1])
                                        os.chmod(FolderOutput + '0/' + NameAndimage_forNewFolder[0], 0o777)

                        except Exception as e:
                            # try:
                            print(str(
                                dt.datetime.now()) + '****************************************************************** Exception  ' + i)

                            pass

                            # cv2.cvtColor(file)

                    f = open(str(AbsPath) + '/' + '2009_S2048.pickle', "wb")
                    f.write(pickle.dumps(data_S2048))
                    f.close()

                    f = open(str(AbsPath) + '/' + '2009_R2048.pickle', "wb")
                    f.write(pickle.dumps(data_R2048))
                    f.close()

                    f = open(str(AbsPath) + '/' + 'ListFilesNames.pickle', "wb")
                    f.write(pickle.dumps(data_ListFilesNames))
                    f.close()



                except Exception as e:
                    print(str(
                        dt.datetime.now()) + ' *******************************************************Exception  Exception   Exception  Exception  Exception  Exception  Exception  Exception  Exception  ')

                # PocerssRecognizing = False

                RenameFoders(model, Glasis_Net, FolderOutput, faceNet1)
            time.sleep(3600)


class Tr_DeleteOldFilesForSecurity(Thread):
    def __init__(self):
        """Инициализация потока"""
        Thread.__init__(self)

    def run(self):
        while True:
            for i, j, y in os.walk(str(AbsPath) + '/' + 'FaceOldBackup/.'):
                if i != 'FaceOldBackup/.':

                    try:
                        dTime = i[i.find('.') + 2:]
                        dtObject = dt.datetime.strptime(dTime, "%d-%m-%Y_%H-%M-%S.%f")
                        now = dt.datetime.now()
                        res = now - datetime.timedelta(days=14)
                        if (dtObject < res):
                            # os.remove( i+'/'+f)
                            shutil.rmtree(i)

                    #       pass

                    except:
                        pass

            time.sleep(3600 * 24)


class Thread_FindFaceForSecurity(Thread):
    def __init__(self):
        """Инициализация потока"""
        Thread.__init__(self)

    def run(self):
        SecurityUnitServer.MainProcFindFaceForSec()


thread_findFaceForSec = Thread_FindFaceForSecurity()

thread_findFaceForSec.start()
thread_2 = Tr_DeleteFolder()
thread_2.start()

thread_3 = tread_FaceRecProc()
thread_3.start()

thread_4 = Tr_DeleteOldFilesForSecurity()
thread_4.start()


class Tr_Cam2(Thread):
    def __init__(self):
        """Инициализация потока"""
        Thread.__init__(self)

    def run(self):
        pass
        #Если нет второй камеры - закоментировать сдесь
        SecurityUnit.StartCam2(IPcam2, Br)


thread_Cam2 = Tr_Cam2()
thread_Cam2.start()


class tread_FolderRenameSecur(Thread):
    def __init__(self):
        """Инициализация потока"""
        Thread.__init__(self)

    def run(self):
        model = load_model(str(AbsPath) + '/' + "mask_detector.model")
        Glasis_Net = load_model(str(AbsPath) + '/' + "Glas_detector.model")
        faceNet1 = cv2.dnn.readNet(str(AbsPath) + '/' + "deploy.prototxt",
                                   str(AbsPath) + '/' + "res10_300x300_ssd_iter_140000.caffemodel")
        while True:
            try:
                RenameFoders(model, Glasis_Net, str(AbsPath) + '/' + 'FaceRecogition_Security/', faceNet1)
                pass

            except:
                pass
            time.sleep(1)


tread_FolderRenameSecur_ = tread_FolderRenameSecur()
tread_FolderRenameSecur_.start()

while True:
    # Capture frame-by-frame

    #try:
    #    cv2.imshow('Video2', SecurityUnit.fr1)
    #    cv2.waitKey(1)
    #except:

    #    pass

    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('Game over')


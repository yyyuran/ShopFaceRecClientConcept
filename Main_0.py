
import os

import align.detect_face as detect_face
import cv2
import imutils
import math
import time
import torch

import numpy as np
import tensorflow as tf
import cpbd as cp
import datetime

import Mtcnn_my

import mediapipe as mp
import argparse
import random as ran

import sys

"""
def trace(frame, event, arg):
    print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace


def test():
    print("Line 8")
    print("Line 9")


sys.settrace(trace)
test()
"""

project_dir = os.path.dirname(os.path.abspath(__file__))
AbsPath = project_dir


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-br', default='Br105')
    parser.add_argument('-IPcam1', default='10.57.7.34')
    parser.add_argument('-IPcam2', default='10.57.7.40')
    parser.add_argument('-IdKassaRmIn1C', default='cam5')
    return parser


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    # print (namespace)

    # print("Привет, {}!".format(namespace.name))

Br = namespace.br
IdKassaRmIn1C=namespace.IdKassaRmIn1C



IPcam1 = namespace.IPcam1


mp_drawing = mp.solutions.drawing_utils
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn_SeNet = Mtcnn_my.MTCNN_(
    image_size=224, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
    device=device
)

from lib.face_utils import judge_side_face
from src.sort import Sort




if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])



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



face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, max_num_faces=1, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
def improve_contrast_image_using_clahe(image) :
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(1, 1))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image

def SaveImageForRec(im, TrackID, FotoSize):
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    #im = cv2.imread(  '111/5.bmp')
    image = im
    # results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to

    # pass by reference.

    image.flags.writeable = False
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
            #cv2.imwrite('Temp/1.bmp', image)
            ###################
            point10y = int(face_landmarks.landmark[10].y * image.shape[0])
            point152y = int(face_landmarks.landmark[152].y * image.shape[0])
            koefizient = (point152y - point10y) / image.shape[0]

            LeftPointX = int(face_landmarks.landmark[133].x * image.shape[1])
            LeftPointY = int(face_landmarks.landmark[133].y * image.shape[0])

            CenterPointX = int(face_landmarks.landmark[168].x * image.shape[1])
            CenterPointY = int(face_landmarks.landmark[168].y * image.shape[0])

            LeftSize = math.sqrt((CenterPointX - LeftPointX) * (CenterPointX - LeftPointX) + (
                    (CenterPointY - LeftPointY) * (CenterPointY - LeftPointY)))

            RightPointX = int(face_landmarks.landmark[362].x * image.shape[1])
            RightPointY = int(face_landmarks.landmark[362].y * image.shape[0])
            RihtSize = math.sqrt((CenterPointX - RightPointX) * (CenterPointX - RightPointX) + (
                    (CenterPointY - RightPointY) * (CenterPointY - RightPointY)))

            NosePointY1 = int(face_landmarks.landmark[168].y * image.shape[0])
            NosePointY2 = int(face_landmarks.landmark[4].y * image.shape[0])
            NosePointX1 = int(face_landmarks.landmark[168].x * image.shape[1])
            NosePointX2 = int(face_landmarks.landmark[4].x * image.shape[1])

            LeftPointY = int(face_landmarks.landmark[133].y * image.shape[0])
            RightPointY = int(face_landmarks.landmark[362].y * image.shape[0])
            AgeY = abs(LeftPointY - RightPointY) / 2.0
            if (LeftPointY > RightPointY):
                AgePointY = RightPointY + AgeY
            else:

                AgePointY = LeftPointY + AgeY
            OneProcent = (abs(RihtSize + LeftSize)) / 100.0
            # LeftPorocents = (abs(CenterPointX - LeftPointX)) / OneProcent
            LeftPorocents = LeftSize / (RihtSize + LeftSize) * 100

            RightPorocents = (abs(RightPointX - CenterPointX)) / OneProcent
            # HieghtNoseProc = (abs(NosePointY1 - NosePointY2)) / OneProcent
            HieghtNoseProc = abs(math.sqrt((NosePointX1 - NosePointX2) * (NosePointX1 - NosePointX2) + (
                    (NosePointY1 - NosePointY2) * (NosePointY1 - NosePointY2)))) / OneProcent
            # im = image
            # FotoSize = im.shape[1]
            # TrackID = 111111
            # print('есть лицо')

            # ListImForThisTracker = []
            # ListImForThisTracker_security = []
            # ListTrackers = []
            # HieghtNoseProc ограничение на движение головы вверх(82 - норм)
            if ((HieghtNoseProc > 87) & (LeftPorocents > 36.0) & (LeftPorocents < 64.0) & (
                    CenterPointY + 8 < AgePointY) & (
                        koefizient > 0.3)) & (FotoSize >= 400):
                # if ((HieghtNoseProc > 82) & (LeftPorocents > 38.0) & (LeftPorocents < 62.0) & (CenterPointY + 5 < AgePointY) & (koefizient > 0.3)) & (FotoSize >= 400):
                # print ("                                                                        CenterPointY +7 "+str(CenterPointY+7)+"     AgePointY    "+str(AgePointY))
                # & (shape[9][1] < im.shape[0])
                # print('CenterPointY '+str(CenterPointY))

                # if FotoSize>420:
                # left_eye = extract_eye_center(shape_, LEFT_EYE_INDICES)
                # right_eye = extract_eye_center(shape_, RIGHT_EYE_INDICES)
                left_eye = [LeftPointX, LeftPointY]
                right_eye = [RightPointX, RightPointY]
                M = get_rotation_matrix(left_eye, right_eye)
                height, width = im.shape[:2]
                im = cv2.warpAffine(im, M, (width, height), flags=cv2.INTER_CUBIC)
                ############################################################################################

                try:
                    # image_for_blur = im_temp[shape_.rect.top():shape_.rect.bottom(),shape_.rect.left():shape_.rect.right()]
                    image_for_blur = im
                    if ((image_for_blur.shape[0] != 0) & (image_for_blur.shape[1] != 0)):

                        w = image_for_blur.shape[0]
                        h = image_for_blur.shape[1]

                        image_for_blur = image_for_blur[int(h * 0.4):int(h * 0.8), int(w * 0.2):int(w * 0.70)]
                        # cv2.imwrite('Temp/' + str(ran.random())+'blurrb.bmp', image_for_blur)
                        fm, fm2 = variance_of_laplacian(image_for_blur)
                    else:
                        fm = 0
                        fm2 = 0
                except:
                    fm = 0
                    fm2 = 0

                ############################################################################################
                if (fm >= 0.00500) & (fm2 >= 12) & (fm < 1.0):
                    # if (fm >= 0.0500) & (fm2 >= 50) & (fm < 1.0):
                    ###корр угла
                    x_aligned_SeNet, prob_SeNet = mtcnn_SeNet(improve_contrast_image_using_clahe(im), return_prob=True)
                    if prob_SeNet is not None:
                        if ((prob_SeNet > 0.5)):

                            # if TraclLast==TrackID:
                            AnglalHorizontalLast
                            # global LatsTrackerID
                            # global data_CountFilesInTracker
                            ListImForThisTracker

                            ListindexTrackers = []
                            for ind in range(len(ListTrackers)):
                                if ListTrackers[ind] == TrackID:
                                    ListindexTrackers.append(ind)
                            if len(ListindexTrackers) != 0:
                                a = AnglalHorizontalLast[max(ListindexTrackers)]
                            else:
                                a = 0
                            # print('                               '+str(int(a))+'  '+str(int(LeftPorocents)))
                            if abs(int(a) - int(LeftPorocents)) > 1:
                                ListImForThisTracker.append(im)
                                ListImForThisTracker_security.append(im)
                                ListTrackers.append(TrackID)
                                AnglalHorizontalLast.append(LeftPorocents)
                                print(
                                    ' кол-во файлов в буфере основного распознования ' + str(len(ListImForThisTracker)))
                                print(' кол-во файлов в буфере secutity ' + str(len(ListImForThisTracker_security)))
                        else:
                            print('Фото бракованая')
                    else:
                        print('Фото бракованая')
                else:

                    print('Блюр 1 ----------------------- ' + str(int(fm * 1000) / 1000) + '_' + str(int(fm2)))
                # else:
                # print("Размер фото мал для основного распозования")
            else:

                if FotoSize < 400:
                    print("Размер фото мал для основного распозования")
                else:
                    print('Отвёрнута голова для основного распознования  ' + str(HieghtNoseProc) + ' > 81   ' + str(
                        CenterPointY + 2) + ' < ' + str(AgePointY))
                # для секурити этот блок
                if ((HieghtNoseProc > 81) & (LeftPorocents > 33.0) & (LeftPorocents < 67.0) & (
                        CenterPointY + 2 < AgePointY) & (koefizient > 0.3)):
                    # & (shape[9][1] < im.shape[0])
                    # print('CenterPointY '+str(CenterPointY))
                    # if FotoSize > 350:
                    # left_eye = extract_eye_center(shape_, LEFT_EYE_INDICES)
                    # right_eye = extract_eye_center(shape_, RIGHT_EYE_INDICES)
                    left_eye = [LeftPointX, LeftPointY]
                    right_eye = [RightPointX, RightPointY]
                    M = get_rotation_matrix(left_eye, right_eye)
                    height, width = im.shape[:2]
                    im = cv2.warpAffine(im, M, (width, height), flags=cv2.INTER_CUBIC)
                    ############################################################################################

                    try:
                        # image_for_blur = im_temp[shape_.rect.top():shape_.rect.bottom(),
                        #                 shape_.rect.left():shape_.rect.right()]
                        image_for_blur = im
                        if ((image_for_blur.shape[0] != 0) & (image_for_blur.shape[1] != 0)):

                            w = image_for_blur.shape[0]
                            h = image_for_blur.shape[1]

                            image_for_blur = image_for_blur[int(h * 0.4):int(h * 0.8), int(w * 0.2):int(w * 0.70)]
                            # cv2.imwrite('Temp/' + str(ran.random()) + 'blurrb.bmp', image_for_blur)
                            fm, fm2 = variance_of_laplacian(image_for_blur)
                        else:
                            fm = 0
                            fm2 = 0
                    except:
                        fm = 0
                        fm2 = 0

                    ############################################################################################
                    if (fm >= 0.00500) & (fm2 >= 12) & (fm < 1.0):
                        # if (fm >= 0.0500) & (fm2 >= 50) & (fm < 1.0):
                        ###корр угла
                        x_aligned_SeNet, prob_SeNet = mtcnn_SeNet(improve_contrast_image_using_clahe(im), return_prob=True)
                        if prob_SeNet is not None:
                            if ((prob_SeNet > 0.5)):

                                # if TraclLast==TrackID:
                                # global AnglalHorizontalLast
                                # global LatsTrackerID
                                # global data_CountFilesInTracker

                                ListindexTrackers = []
                                for ind in range(len(ListTrackers)):
                                    if ListTrackers[ind] == TrackID:
                                        ListindexTrackers.append(ind)
                                if len(ListindexTrackers) != 0:
                                    a = AnglalHorizontalLast[max(ListindexTrackers)]
                                else:
                                    a = 0
                                # print('                               ' + str(int(a)) + '  ' + str(
                                #   int(LeftPorocents)))
                                if abs(int(a) - int(LeftPorocents)) > 1:
                                    ListImForThisTracker_security.append(im)
                                    ListImForThisTracker.append(None)
                                    ListTrackers.append(TrackID)
                                    AnglalHorizontalLast.append(LeftPorocents)
                                    print(' кол-во файлов в буфере security ' + str(len(ListImForThisTracker_security)))
                            else:
                                print('Фото бракованая')
                        else:
                            print('Фото бракованая')
                    else:

                        print('Блюр 1 ----------------------- ' + str(int(fm * 1000) / 1000) + '_' + str(
                            int(fm2)))

                else:
                    print('Отвёрнута голова для любого распознования ')
    else:
        print('Не нашёл голову')


AnglalHorizontalLast = [0]
FaceIDcounter = 0
gpuOpt = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, visible_device_list='0', allow_growth=True)
tracker = Sort()


def sendFoto(t):
    print(str(len(ListImForThisTracker)))

    # if ((LatsTrackerID != int(TrackID)) & (LatsTrackerID != 0))or(TrackID==0):
    # if len(ListImForThisTracker) > 2:
    # FaceID = int(ran.random() * 1000000)

    # for f in ListImForThisTracker:
    ListGroupedTrackers = []
    for elem in ListTrackers:
        if elem not in ListGroupedTrackers:
            ListGroupedTrackers.append(elem)
    TreckersForOnlyMainRec = []
    for tr in (ListGroupedTrackers):

        sumEl = 0
        for ind in range(len(ListTrackers)):
            if ListTrackers[ind] == tr:
                try:
                    imageforsend = ListImForThisTracker[ind]
                    if not (imageforsend is None):
                        sumEl = sumEl + 1
                except:
                    pass

        if sumEl > 12:
            TreckersForOnlyMainRec.append(tr)
            FaceID = int(ran.random() * 1000000)
            now = datetime.datetime.now()
            dt_string2 = now.strftime("%d-%m-%Y_%H-%M-%S.%f")
            from pathlib import Path
            Path(str(AbsPath) + '/' + 'DISTR/FaceRecogition_temp/' + dt_string2).mkdir(mode=0o755, parents=True,
                                                                                       exist_ok=True)
            os.chmod(str(AbsPath) + '/' + 'DISTR/FaceRecogition_temp/' + dt_string2, 0o777)
            x = 0
            for ind in range(len(ListTrackers)):
                if ListTrackers[ind] == tr:
                    if x > 89: break
                    try:
                        imageforsend = ListImForThisTracker[ind]
                        now = datetime.datetime.now()
                        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S.%f")
                        cv2.imwrite(str(
                            AbsPath) + '/' + 'DISTR/FaceRecogition_temp/' + dt_string2 + '/' + Br + '_'+IdKassaRmIn1C+'_' + dt_string + '_Tr' + str(
                            FaceID) + '.bmp', imageforsend)
                        os.chmod(str(
                            AbsPath) + '/' + 'DISTR/FaceRecogition_temp/' + dt_string2 + '/' + Br + '_'+IdKassaRmIn1C+'_' + dt_string + '_Tr' + str(
                            FaceID) + '.bmp', 0o777)
                        x = x + 1
                    except:
                        pass
        else:
            print('колво фоток общего распознования меньше 15 на оин треккер')

    for tr in (ListGroupedTrackers):

        sumEl = 0
        for ind in range(len(ListTrackers)):
            if ListTrackers[ind] == tr:
                try:
                    imageforsend = ListImForThisTracker_security[ind]
                    sumEl = sumEl + 1
                except:
                    pass
        if sumEl > 12:

            FaceID = int(ran.random() * 1000000)
            now = datetime.datetime.now()
            dt_string2 = now.strftime("%d-%m-%Y_%H-%M-%S.%f")
            from pathlib import Path
            Path(str(AbsPath) + '/' + 'FaceRecogition_Security/' + dt_string2).mkdir(mode=0o755, parents=True,
                                                                                     exist_ok=True)
            os.chmod(str(AbsPath) + '/' + 'FaceRecogition_Security/' + dt_string2, 0o777)

            Path(str(AbsPath) + '/' + 'FaceOldBackup/' + dt_string2).mkdir(mode=0o755, parents=True, exist_ok=True)
            os.chmod(str(AbsPath) + '/' + 'FaceOldBackup/' + dt_string2, 0o777)
            # if not (tr in TreckersForOnlyMainRec):
            #    Path('DISTR/FaceRecogition_temp_adv_1/' + dt_string2).mkdir(mode=0o755, parents=True, exist_ok=True)
            #    os.chmod('DISTR/FaceRecogition_temp_adv_1/' + dt_string2, 0o777)

            Path(str(AbsPath) + '/' + 'DISTR/FaceRecogition_temp_adv_1/' + dt_string2).mkdir(mode=0o755, parents=True,
                                                                                             exist_ok=True)
            os.chmod(str(AbsPath) + '/' + 'DISTR/FaceRecogition_temp_adv_1/' + dt_string2, 0o777)

            x = 0
            for ind in range(len(ListTrackers)):
                if ListTrackers[ind] == tr:
                    if x > 89: break
                    try:
                        imageforsend = ListImForThisTracker_security[ind]
                        now = datetime.datetime.now()
                        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S.%f")

                        cv2.imwrite(str(
                            AbsPath) + '/' + 'FaceRecogition_Security/' + dt_string2 + '/' + Br + '_'+IdKassaRmIn1C+'_' + dt_string + '_Tr' + str(
                            FaceID) + '.bmp', imageforsend)
                        os.chmod(str(
                            AbsPath) + '/' + 'FaceRecogition_Security/' + dt_string2 + '/' + Br + '_'+IdKassaRmIn1C+'_' + dt_string + '_Tr' + str(
                            FaceID) + '.bmp', 0o777)

                        cv2.imwrite(
                            str(
                                AbsPath) + '/' + 'FaceOldBackup/' + dt_string2 + '/' + Br + '_'+IdKassaRmIn1C+'_' + dt_string + '_Tr' + str(
                                FaceID) + '.bmp', imageforsend)
                        os.chmod(
                            str(
                                AbsPath) + '/' + 'FaceOldBackup/' + dt_string2 + '/' + Br + '_'+IdKassaRmIn1C+'_' + dt_string + '_Tr' + str(
                                FaceID) + '.bmp', 0o777)
                        # if not (tr in TreckersForOnlyMainRec):
                        #    cv2.imwrite('DISTR/FaceRecogition_temp_adv_1/' + dt_string2 + '/'+Br+'_'+IdKassaRmIn1C+'_' + dt_string + '_Tr' + str(FaceID) + '.bmp', imageforsend)
                        #    os.chmod('DISTR/FaceRecogition_temp_adv_1/' + dt_string2 + '/'+Br+'_'+IdKassaRmIn1C+'_' + dt_string + '_Tr' + str(FaceID) + '.bmp', 0o777)

                        cv2.imwrite(str(
                            AbsPath) + '/' + 'DISTR/FaceRecogition_temp_adv_1/' + dt_string2 + '/' + Br + '_'+IdKassaRmIn1C+'_' + dt_string + '_advAngl_Tr' + str(
                            FaceID) + '.bmp', imageforsend)
                        os.chmod(str(
                            AbsPath) + '/' + 'DISTR/FaceRecogition_temp_adv_1/' + dt_string2 + '/' + Br + '_'+IdKassaRmIn1C+'_' + dt_string + '_advAngl_Tr' + str(
                            FaceID) + '.bmp', 0o777)
                        x = x + 1
                    except:
                        pass
            # RenameFoders(model,Glasis_Net,'FaceRecogition_Security/',faceNet1)
        else:
            print('колво фоток security меньше 15 на оин треккер')

        # now = datetime.datetime.now()
        # dt_string = now.strftime("%d-%m-%Y_%H-%M-%S.%f")
    # cv2.imwrite('DISTR/FaceRecogition/' + ''+Br+'_'+IdKassaRmIn1C+'_' + dt_string + '_Tr' + str(FaceID) + '.bmp', f)
    # cv2.imwrite(
    #     'DISTR/FaceRecogition_old/' + ''+Br+'_'+IdKassaRmIn1C+'_' + dt_string + '_Tr' + str(FaceID) + '.bmp', f)
    # else:
    #    print('Файлов в треккере меньше 3 - недопустимо!')

    ListImForThisTracker.clear()
    ListImForThisTracker_security.clear()
    ListTrackers.clear()






with tf.Graph().as_default():
    with tf.Session(config=tf.ConfigProto(gpu_options=gpuOpt,
                                          log_device_placement=False)) as sess:
        pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(project_dir, "align"))
        minsize = 80  # minimum size of face for mtcnn to detect
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        directoryname = '1'
        while True:
            cap = cv2.VideoCapture('rtsp://'+IPcam1+':554/user=admin&password=gj94hp8z&channel=1&stream=0?.sdp')
            while cap.isOpened():
                success, frame_inp = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    cap = cv2.VideoCapture('rtsp://'+IPcam1+':554/user=admin&password=gj94hp8z&channel=1&stream=0?.sdp')
                    continue
                frame = imutils.resize(frame_inp, width=640)
                # SaveImageForRec(image, 1, FotoSize=410)
                try:
                    # frame=improve_contrast_image_using_clahe(frame)
                    if FaceIDcounter > 1000000:
                        # global tracker
                        tracker = None
                        tracker = Sort()
                        FaceIDcounter = 0
                        # FaceIDcounter = int(r.random() * 1000000)

                    # frame1 = imutils.resize(frame, width=640)
                    # boxes = detectFaceOpenCVDnn(net, frame1)
                    #  if len(boxes[1]) > 0:
                    # SaveImageForRec(frame)
                    # global counter

                    final_faces = []
                    addtional_attribute_list = []
                    # ret, frame_inp = cam.read()

                    # frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)

                    r_g_b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # if counter % detect_interval == 0:
                    img_size = np.asarray(frame.shape)[0:2]
                    # mtcnn_starttime = time()
                    faces, points = detect_face.detect_face(r_g_b_frame, minsize, pnet, rnet, onet, threshold,
                                                            factor)

                    # logger.info("MTCNN detect face cost time : {} s".format(
                    #    round(time() - mtcnn_starttime, 3)))  # mtcnn detect ,slow
                    face_sums = faces.shape[0]
                    if face_sums > 0:
                        face_list = []
                        for i, item in enumerate(faces):
                            score = round(faces[i, 4], 6)
                            if score > face_score_threshold:
                                det = np.squeeze(faces[i, 0:4])

                                # face rectangle
                                det[0] = np.maximum(det[0] - margin, 0)
                                det[1] = np.maximum(det[1] - margin, 0)
                                det[2] = np.minimum(det[2] + margin, img_size[1])
                                det[3] = np.minimum(det[3] + margin, img_size[0])
                                face_list.append(item)

                                # face cropped
                                bb = np.array(det, dtype=np.int32)

                                facial_landmarks = []
                                for j in range(5):
                                    # item = [tolist[j], tolist[(j + 5)]]
                                    facial_landmarks.append(item)

                                cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :].copy()

                                dist_rate, high_ratio_variance, width_rate = judge_side_face(
                                    np.array(facial_landmarks))

                                # face addtional attribute(index 0:face score; index 1:0 represents front face and 1 for side face )
                                item_list = [cropped, score, dist_rate, high_ratio_variance, width_rate]
                                addtional_attribute_list.append(item_list)

                        final_faces = np.array(face_list)
                    # print(str(datetime.datetime.now()))
                    trackers = tracker.update(final_faces, img_size, directoryname, addtional_attribute_list,
                                              detect_interval)

                    # counter += 1
                    if ((len(trackers) == 0) & (
                            ((len(ListImForThisTracker) > 0)) or (len(ListImForThisTracker_security) > 0))):
                        sendFoto(0)
                        #face_mesh.close()
                        #face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=1,
                        #                                  min_tracking_confidence=0.5)

                    for d in trackers:
                        #  if not no_display:
                        d = d.astype(np.int32)
                        FaceIDcounter = d[4]
                        cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colours[d[4] % 32, :] * 255, 3)
                        # cv2.rectangle(frame, (d[0], d[1]), (d[0]+10, d[3]+10), colours[d[4] % 32, :] * 255, 3)
                        if final_faces != []:
                            cv2.putText(frame, 'ID : %d ' % (d[4]), (d[0] - 10, d[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.75,
                                        colours[d[4] % 32, :] * 255, 2)
                            # cv2.putText(frame, 'DETECTOR', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            #            (1, 1, 1), 2)
                        else:
                            cv2.putText(frame, 'ID : %d' % (d[4]), (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.75,
                                        colours[d[4] % 32, :] * 255, 2)

                        y1 = int(d[1] / k_h)
                        y2 = int(d[3] / k_h)

                        x1 = int(d[0] / k_W)
                        x2 = int(d[2] / k_W)

                        x1 = x1 - 30
                        if x1 < 0:
                            x1 = 0
                        y1 = y1 - 30
                        if y1 < 0:
                            y1 = 0

                        x2 = x2 + 30
                        if x2 > 1920:
                            x2 = 1920
                        y2 = y2 + 30
                        if y2 > 1080:
                            y2 = 1080

                        face = frame_inp[y1: y2, x1: x2]

                        now = datetime.datetime.now()
                        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S.%f")
                        # print (str('   x1  '+str(x1))+'  x2  '+str(x2))
                        if (x1 > 180) & (x2 < 1740):
                            # cv2.imwrite('Temp/' + ''+Br+'_cam2_' + dt_string + '_' + str(d[4]) + '.bmp', face)
                            # if (abs(y2-y1) > 420):
                            if (abs(y2 - y1) > 350):
                                SaveImageForRec(face, str(d[4]), FotoSize=abs(y2 - y1))
                            else:
                                print('Размер фото мал для любого распознования')
                    #fr = frame
                    # else:
                    #    time.sleep(0.001)
                    # global fr1
                    # fr1= frame1
                    # cv2.imshow('Video1', frame1)
                    # cv2.waitKey(1)
                except:
                    # vs.release()
                    # vs = cv2.VideoCapture(0)
                    # vs = cv2.VideoCapture('rtsp://' + IPcam1 + ':554/user=admin&password=gj94hp8z&channel=1&stream=0?.sdp')
                    print('Error')
                    time.sleep(0.1)

                #cv2.imshow('MediaPipe FaceMesh', frame)
                #if cv2.waitKey(5) & 0xFF == 27:
                #    break
            print('Что то с камерой')
print('Game over')
face_mesh.close()
cap.release()


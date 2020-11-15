# 导入dlib库以及依赖
import dlib
import cv2
import numpy as np
from scipy.spatial import distance
import os
from imutils import face_utils
from scipy.spatial import distance as dist

# 主函数入口
if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()  # face_detector

    predictor = dlib.shape_predictor(
        "E://BaiduNetdiskDownload//dataandmodel4dlib//data4dlib//shape_predictor_68_face_landmarks.dat")  # face_feature_point detector,using absolute path

    # define parameters
    EYE_AR_THRESH = 0.3  # EAR Threshold
    # When the ear threshold is less than a certain number of consecutive frames
    EYE_AR_CONSEC_FRAMES = 3

    # Serial number of corresponding feature points
    RIGHT_EYE_START = 37 - 1
    RIGHT_EYE_END = 42 - 1
    LEFT_EYE_START = 43 - 1
    LEFT_EYE_END = 48 - 1

    FACIAL_LANDMARKS_68_IDXS = dict([
        ("mouth", (48, 68)),
        ("right_eyebrow", (17, 22)),
        ("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 36)),
        ("jaw", (0, 17))
    ])
'''
EYE_AR_THRESH是判断阈值，默认为0.3。如果EAR大于它，则认为眼睛是睁开的；如果EAR小于它，则认为眼睛是闭上的。
EYE_AR_CONSEC_FRAMES表示的是，当EAR小于阈值时，接连多少帧一定发生眨眼动作。只有小于阈值的帧数超过了这个值时，
才认为当前眼睛是闭合的，即发生了眨眼动作；否则则认为是误操作。
RIGHT_EYE_START、RIGHT_EYE_END、LEFT_EYE_START、LEFT_EYE_END：这几个都对应了人脸特征点中对应眼睛的那几个特征点的序号。
由于list中默认从0开始，为保持一致，所以减一。
'''


#  EAR（eye aspect ratio）计算函数

def eye_aspect_ratio(eye):
    # 计算距离，竖直的
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算距离，水平的
    C = dist.euclidean(eye[0], eye[3])
    # ear值
    ear = (A + B) / (2.0 * C)
    return ear


# 处理视频流
frame_counter = 0  # 连续帧计数
blink_counter = 0  # 眨眼计数
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 更改API设置 初次运行时报出cv::color empty exception ,将参数改为0，调用本机摄像头，问题解决
while 1:
    ret, img = cap.read()  # 读取视频流的一帧

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转成灰度图像
    rects = detector(gray, 0)  # 人脸检测
    for rect in rects:  # 遍历每一个人脸
        print('-' * 20)
        shape = predictor(gray, rect)  # 检测特征点
        points = face_utils.shape_to_np(shape)  # convert the facial landmark (x, y)-coordinates to a NumPy array
        leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]  # 取出左眼对应的特征点
        rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]  # 取出右眼对应的特征点
        leftEAR = eye_aspect_ratio(leftEye)  # 计算左眼EAR
        rightEAR = eye_aspect_ratio(rightEye)  # 计算右眼EAR
        print('leftEAR = {0}'.format(leftEAR))
        print('rightEAR = {0}'.format(rightEAR))

        ear = (leftEAR + rightEAR) / 2.0  # 求左右眼EAR的均值

        leftEyeHull = cv2.convexHull(leftEye)  # 寻找左眼轮廓
        rightEyeHull = cv2.convexHull(rightEye)  # 寻找右眼轮廓
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)  # 绘制左眼轮廓
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)  # 绘制右眼轮廓

        # 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
        if ear < EYE_AR_THRESH:
            frame_counter += 1
        else:
            if frame_counter >= EYE_AR_CONSEC_FRAMES:
                blink_counter += 1
            frame_counter = 0

        # 在图像上显示出眨眼次数blink_counter和EAR
        cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

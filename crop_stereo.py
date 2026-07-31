import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import Calibrator

calibratorL = Calibrator() 
calibratorL.setCheckerboard((8, 5), False)
calibratorR = Calibrator() 
calibratorR.setCheckerboard((8, 5), False)

#we considered cheeseboard which have 8 corners vertically and 5 corners horizontally
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((8 * 5, 3), np.float32)
objp[:, :2] = np.mgrid[0: 8, 0: 5].T.reshape(-1, 2)
img_ptsL = []
img_ptsR = []
obj_pts = []
# pathL = "left/"
# pathR = "right/"

video = cv2.VideoCapture('20221227_140406.mp4')
#read left and right camera images
#find corners
#draw corners
while True:
    ret, frame = video.read()
    if not ret:
        break
    imgL = frame[:, frame.shape[1] // 2:, :]
    imgR = frame[:, : frame.shape[1] // 2, :]
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    outputL=imgL.copy()
    outputR=imgR.copy()
    
    errR, imgR, cornersR = calibratorR.detectCorner(outputR)# cv2.findChessboardCorners(outputR,(8, 5),None)
    errL, imgL, cornersL = calibratorL.detectCorner(outputL)#cv2.findChessboardCorners(outputL,(8, 5),None)

    if errL == 0 and errR == 0:
        obj_pts.append(objp)
        # disp_images = np.hstack((imgL_corner, imgR_corner))
        # cv2.cornerSubPix(imgR_gray,cornersR,(11, 11),(-1, -1),criteria)
        # cv2.cornerSubPix(imgL_gray,cornersL,(11, 11),(-1, -1),criteria)
        # cv2.drawChessboardCorners(outputR, (8, 5), cornersR, retR)
        # cv2.drawChessboardCorners(outputL, (8, 5), cornersL, retL)

        
        img_ptsL.append(cornersL)
        img_ptsR.append(cornersR)
    # else:
    disp_images = np.hstack((imgR, imgL))
    cv2.imshow('disp', disp_images)
    # cv2.imshow('cornersL', imgL_corner)
    cv2.waitKey(1)
    
cv2.destroyAllWindows()
video.release()   

#intrinsic calibrate
# calibratorR.calibrate()
# calibratorL.calibrate()

# new_mtxL = calibratorL.getCameraMatrix()
# distL = calibratorL.getDistortionCoeffs()
# new_mtxR = calibratorR.getCameraMatrix()
# distR = calibratorR.getDistortionCoeffs()

#combined two new matrix and apply stereo calibrate 
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts,
                                                          img_ptsL,
                                                          img_ptsR,
                                                          new_mtxL,
                                                          distL,
                                                          new_mtxR,
                                                          distR,
                                                          imgL_gray.shape[::-1],
                                                          criteria_stereo,
                                                          flags)


#rectify new matrix
rectify_scale= 1
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR,
                                                 imgL_gray.shape[::-1], Rot, Trns)
 
#get undistorted rectify map for left images
Left_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                             imgL_gray.shape[::-1], cv2.CV_16SC2)
#get undistorted rectify map for right images                                             
Right_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                              imgR_gray.shape[::-1], cv2.CV_16SC2)
                                                                         
                                            



#save undistorted parameters for both images
print("Saving paraMeters ......")
cv_file = cv2.FileStorage("improved_params.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
cv_file.write("Rotation", Rot)
cv_file.write("Translation", Trns)
cv_file.release()




# #apply those points in real life scene
# CamL= cv2.VideoCapture(0)
# CamR= cv2.VideoCapture(1)

print("Reading parameters ......")
cv_file = cv2.FileStorage("improved_params.xml", cv2.FILE_STORAGE_READ)

Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()
video = cv2.VideoCapture('20221227_140406.mp4')

while True:
    ret, frame = video.read()
    if not ret:
        break
    imgL = frame[:, frame.shape[1] // 2:, :]
    imgR = frame[:, : frame.shape[1] // 2, :]
    
    imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)

    Left_nice= cv2.remap(imgL,Left_Stereo_Map_x,Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice= cv2.remap(imgR,Right_Stereo_Map_x,Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    output = Right_nice.copy()
    output[:,:,0] = Right_nice[:, :, 0]
    output[:,:,1] = Right_nice[:, :, 1]
    output[:,:,2] = Left_nice[:, :, 2]

    # output = Left_nice+Right_nice
    # output = cv2.resize(output,(700,700))
    cv2.namedWindow("3D movie", cv2.WINDOW_NORMAL)
    cv2.imshow("3D movie", output)

    cv2.waitKey(1)
    
    # else:
    #     break
cv2.destroyAllWindows()
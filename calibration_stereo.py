import cv2
import argparse
from utils import Calibrator
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", help="Input source (camera index/video path)", default='0')
    parser.add_argument("-c", "--checkerboard_size", help="Checkboard size (number of columns x number of rows) (default is 6,4)", type=lambda s: [int(item) for item in s.split(',')], default = [6, 4])
    parser.add_argument("-l", "--stereo_left", help="Left camera calibration file path", default='stereo_left.xml', type=str)
    parser.add_argument("-r", "--stereo_right", help="Right camera calibration file path", default='stereo_right.xml', type=str)
    parser.add_argument("-o", "--output", help="Output calibration file path", default='stereo.xml')
    parser.add_argument("-t", "--thermal", help="Is thermal camera (default is false)", action='store_true')
    parser.add_argument("--low_res", help="Is low-resolution stream", action='store_true')
    args = parser.parse_args()
    
    calibratorL = Calibrator() 
    calibratorL.getCalibInfo(args.stereo_left)
    calibratorR = Calibrator() 
    calibratorR.getCalibInfo(args.stereo_right)
    
    nCols = max(args.checkerboard_size[0], args.checkerboard_size[1])
    nRows = min(args.checkerboard_size[0], args.checkerboard_size[1])
    calibratorL.setCheckerboard((nCols, nRows), args.low_res)
    calibratorR.setCheckerboard((nCols, nRows), args.low_res)
    #we considered cheeseboard which have 8 corners vertically and 5 corners horizontally
    objp = np.zeros((nCols * nRows, 3), np.float32)
    objp[:, :2] = np.mgrid[0: nCols, 0: nRows].T.reshape(-1, 2)
    img_ptsL = []
    img_ptsR = []
    obj_pts = []
    
    imgL_gray = None
    
    source = args.index
    cam = cv2.VideoCapture(eval(source) if source.isnumeric() else source)
    if not source.isnumeric():
        frameNum = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    while True:
        haveFrame, frame = cam.read()
        
        if not haveFrame:
            if source.isnumeric() or cam.get(cv2.CAP_PROP_POS_FRAMES) >= frameNum:
                break
            else:
                continue
        
        imgL = frame[:, frame.shape[1] // 2:, :]
        imgR = frame[:, : frame.shape[1] // 2, :]
        
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        
        outputL=imgL.copy()
        outputR=imgR.copy()
    
        errR, imgR, cornersR = calibratorR.detectCorners(outputR, False)
        errL, imgL, cornersL = calibratorL.detectCorners(outputL, False)

        if errL == 0 and errR == 0:
            obj_pts.append(objp)
            img_ptsL.append(cornersL)
            img_ptsR.append(cornersR)
        
        disp_images = np.hstack((imgR, imgL))
        cv2.imshow('disp', disp_images)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

    new_mtxL = calibratorL.getCameraMatrix()
    distL = calibratorL.getDistortionCoeffs()
    new_mtxR = calibratorR.getCameraMatrix()
    distR = calibratorR.getDistortionCoeffs()

    #combined two new matrix and apply stereo calibrate 
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    
    print("Calibrating...")
    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts,
                                                            img_ptsL,
                                                            img_ptsR,
                                                            new_mtxL,
                                                            distL,
                                                            new_mtxR,
                                                            distR,
                                                            None,
                                                            criteria_stereo,
                                                            flags)
    #rectify new matrix
    rectify_scale = 1
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR,
                                                    imgL_gray.shape[::-1], Rot, Trns)
    
    #get undistorted rectify map for left images
    Left_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                                imgL_gray.shape[::-1], cv2.CV_16SC2)
    #get undistorted rectify map for right images                                             
    Right_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                                imgR_gray.shape[::-1], cv2.CV_16SC2)

    #save undistorted parameters for both images
    print("Saving parameters ......")
    cv_file = cv2.FileStorage(args.output, cv2.FILE_STORAGE_WRITE)
    cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
    cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
    cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
    cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
    cv_file.write("Rotation", Rot)
    cv_file.write("Translation", Trns)
    cv_file.release()
    print('Finished!')
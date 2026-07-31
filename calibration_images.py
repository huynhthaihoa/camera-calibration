import cv2
import argparse
import os
from glob import glob
from utils import Calibrator

import numpy as np
CLAHE = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8, 8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input directory", required=True)
    parser.add_argument("-c", "--checkerboard_size", help="Checkboard size (number of columns x number of rows) (default is 6,4)", type=lambda s: [int(item) for item in s.split(',')], default = [6, 4])
    parser.add_argument("-f", "--fisheye", help="Is fisheye camera", action='store_true')
    parser.add_argument("-t", "--thermal", help="Is thermal camera (default is false)", action='store_true')
    parser.add_argument("-l", "--low_res", help="Is low-resolution stream", action='store_true')
    parser.add_argument("-p", "--param_thres", help="Param threshold for evaluating good detected corners (default is 0.2)",
                    type=float, default=0.2)
    parser.add_argument("-q", "--quantity_thres", help="Minimum number of good images for calibration (default is 40)",
                    type=int, default=10)
    parser.add_argument("-o", "--output_path", help="Path to the output OpenCV calibration file (.xml file)",
                    type=str, default='output.xml')
    parser.add_argument("-g", "--good", help="Directory to save good images", type=str, default="")
    parser.add_argument("-d", "--detect", help="Directory to save detect images", type=str, default="")
    
    args = parser.parse_args()
    print(args)
    # source = args.index
    # cam = cv2.VideoCapture(eval(source) if source.isnumeric() else source)
    imgPaths = glob(args.input + "/*.png") + glob(args.input + "/*.jpg")
    OUTPUT_FILE = args.output_path
    calibrator = Calibrator(args.fisheye, args.thermal, args.param_thres, args.quantity_thres) 
    calibrator.setCheckerboard(args.checkerboard_size, args.low_res) # args.thermal or 
    
    good = args.good 
    if good != "" and os.path.isdir(good) is False:
        os.mkdir(good)
        
    detect = args.detect 
    if detect != "" and os.path.isdir(detect) is False:
        os.mkdir(detect)
    
    for imgPath in imgPaths:
        frame = cv2.imread(imgPath)
        raw = frame.copy()

        name = os.path.basename(imgPath)
        err, _, img, _ = calibrator.detectCorners(frame)#, args.thermal)

        label = "Accumulating..."
        n = calibrator.getAccumulatedLen()
        if err != -1:
            if good != "":
                # gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                # gray = np.array(255 - gray, dtype=np.uint8)
                # gray = CLAHE.apply(gray)
                cv2.imwrite("{}/{}".format(good, name), raw)
            if detect != "":
                result = img.copy()
                cv2.imwrite("{}/{}".format(detect, name), img)
        if not err:
            print("Detected good corners. Accumulated {} images.".format(n)) 
            # if good != "":
            #     cv2.imwrite("{}/{}".format(good, name), raw)
            # if detect != "":
            #     result = img.copy()
            #     cv2.imwrite("{}/{}".format(detect, name), img)
        elif err == 1:
            print("Detected bad corners. Accumulated {} images.".format(n)) 
        else:
            print("Detect corners failed. Accumulated {} images".format(n))
        if calibrator.isGoodEnough():
            print("Enough image for calibration!")
        
        cv2.imshow("Result", img)
            
        if cv2.waitKey(1) == ord('q'):# and calibrator.goodenough:
            break
    
    if calibrator.isGoodEnough():
        calibrator.calibrate()
        calibrator.saveCalib(OUTPUT_FILE)
    else:
        print("Not enough images for calibration!")




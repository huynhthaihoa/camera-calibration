import cv2
import argparse
import os
from utils import Calibrator
import numpy as np

def renderProgress(image, progress, isGoodEnough):
    frame = np.ones((image.shape[0], image.shape[1] + 310, 3), dtype=np.uint8) * 255
    frame[:image.shape[0], :image.shape[1], :] = image.copy()
    
    cv2.putText(frame, 'X: {:.3f}%'.format(progress[0] * 100), (image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.putText(frame, 'Y: {:.3f}%'.format(progress[1] * 100), (image.shape[1] + 10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.putText(frame, 'Size: {:.3f}%'.format(progress[2] * 100), (image.shape[1] + 10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.putText(frame, 'Skew: {:.3f}%'.format(progress[3] * 100), (image.shape[1] + 10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    
    if int(progress[0]) == 1:
        cv2.line(frame, (image.shape[1] + 10, 40), (image.shape[1] + 10 + 200, 40), (0, 255, 0), 4)
    else:
         cv2.line(frame, (image.shape[1] + 10, 40), (image.shape[1] + 10 + int(progress[0] * 200), 40), (0, 255, 255), 4)
    
    if int(progress[1]) == 1:
        cv2.line(frame, (image.shape[1] + 10, 110), (image.shape[1] + 10 + 200, 110), (0, 255, 0), 4)
    else:
         cv2.line(frame, (image.shape[1] + 10, 110), (image.shape[1] + 10 + int(progress[1] * 200), 110), (0, 255, 255), 4)

    if int(progress[2]) == 1:
        cv2.line(frame, (image.shape[1] + 10, 180), (image.shape[1] + 10 + 200, 180), (0, 255, 0), 4)
    else:
         cv2.line(frame, (image.shape[1] + 10, 180), (image.shape[1] + 10 + int(progress[2] * 200), 180), (0, 255, 255), 4)

    if int(progress[3]) == 1:
        cv2.line(frame, (image.shape[1] + 10, 250), (image.shape[1] + 10 + 200, 250), (0, 255, 0), 4)
    else:
         cv2.line(frame, (image.shape[1] + 10, 250), (image.shape[1] + 10 + int(progress[3] * 200), 250), (0, 255, 255), 4)
    
    if isGoodEnough:
        text = 'Accumulated enough images. Press Q to calibrate!'
    else:
        text = 'Accumulating images...'
    
    cv2.putText(frame, text, (image.shape[1] + 10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    
    return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", help="Input source (camera index/video path)", default='0')
    parser.add_argument("-s", "--stereo", help="Stereo mode (-1: left, 0: center, 1: right)", type=int, default=0)
    parser.add_argument("-c", "--checkerboard_size", help="Checkboard size (number of corners per row x number of corners per col) (default is 6,4)", type=lambda s: [int(item) for item in s.split(',')], default = [7, 4])
    parser.add_argument("-f", "--fisheye", help="Is fisheye camera", action='store_true')
    parser.add_argument("-t", "--thermal", help="Is thermal camera (default is false)", action='store_true')
    parser.add_argument("-l", "--low_res", help="Is low-resolution stream", action='store_true')
    parser.add_argument("-p", "--param_thres", help="Param threshold for evaluating good detected corners (default is 0.2)",
                    type=float, default=0.2)
    parser.add_argument("-q", "--quantity_thres", help="Minimum number of good images for calibration (default is 40)",
                    type=int, default=40)
    parser.add_argument("-o", "--output_path", help="Path to the output OpenCV calibration file (.xml file)",
                    type=str, default='output.xml')
    parser.add_argument("-g", "--good", help="Directory to save good images", type=str, default="")
    parser.add_argument("-d", "--detect", help="Directory to save detect images", type=str, default="")

    args = parser.parse_args()
    print(args)
    source = args.index
    if source.isnumeric():
        cam = cv2.VideoCapture(eval(source))#, cv2.CAP_DSHOW)
        # w,h = 640, 480 
        # cam.set(3, w)
        # cam.set(4, h)
    else:
        cam = cv2.VideoCapture(source)
    # cam = cv2.VideoCapture(eval(source) if source.isnumeric() else source)
    OUTPUT_FILE = args.output_path
    calibrator = Calibrator(args.fisheye, args.thermal, args.param_thres, args.quantity_thres) 
    calibrator.setCheckerboard(args.checkerboard_size, args.low_res) # args.thermal or 
    
    good = args.good 
    if good != "" and os.path.isdir(good) is False:
        os.mkdir(good)
        
    detect = args.detect 
    if detect != "" and os.path.isdir(detect) is False:
        os.mkdir(detect)
    
    if not source.isnumeric():
        frameNum = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    idx = 0
    while True:
        haveFrame, frame = cam.read()
        if not haveFrame:
            if source.isnumeric() or cam.get(cv2.CAP_PROP_POS_FRAMES) >= frameNum:
                break
            else:
                continue
        
        if args.stereo == -1:
            frame = frame[:, frame.shape[1] // 2 :, :]
        elif args.stereo == 1:
            frame = frame[:, : frame.shape[1] // 2, :]        
        raw = frame.copy()
        err, _, img, progress = calibrator.detectCorners(frame)

        label = "Accumulating..."
        n = calibrator.getAccumulatedLen()
        # if success:
        #     if calibrator.isGoodEnough():
        #         print("Detect corners successfully. Accumulated {} images. Enough images for calibration!".format(calibrator.getAccumulatedLen()))
        #     else:
        #         print("Detect corners successfully. Accumulated {} images".format(calibrator.getAccumulatedLen()))
        if not err:
            print("Detected good corners. Accumulated {} images. Progress: {}".format(n, progress)) 
            if good != "":
                cv2.imwrite("{}/img{}.jpg".format(good, idx), raw)
            if detect != "":
                result = img.copy()
                cv2.imwrite("{}/img{}.jpg".format(detect, idx), result)
            idx += 1

        elif err == 1:
            print("Detected bad corners. Accumulated {} images. Progress: {}".format(n, progress)) 
        else:
            print("Detect corners failed. Accumulated {} images. Progress: {}".format(n, progress))
        if calibrator.isGoodEnough():
            print("Enough image for calibration!")
        
        progress_frame = renderProgress(img, progress, calibrator.isGoodEnough())
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Result", progress_frame)
        
        # idx += 1
        
        if cv2.waitKey(1) == ord('q'):# and calibrator.goodenough:
            break
    
    if calibrator.isGoodEnough():
        calibrator.calibrate()
        calibrator.saveCalib(OUTPUT_FILE)
    else:
        print("Not enough images for calibration!")




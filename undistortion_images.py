import cv2
import argparse
from utils import Calibrator
import os
from glob import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input directory", required=True)
    parser.add_argument("-f", "--horizontal_flip", help="horizontal flip", action="store_true")
    parser.add_argument("-v", "--vertical_flip", help="vertical flip", action="store_true")

    parser.add_argument("-o", "--output", help="Output directory", required=True)
    # parser.add_argument("-m", "--model", help="Camera model (0: pinhole (default), 1: fisheye)", type=int, default=0)
    parser.add_argument("-c", "--calibration_path", help="Path to the OpenCV calibration file (.xml file)",
                    type=str, required=True)
    parser.add_argument("-z", "--zoom", help="Zoom value, \
        range from 0 (zoomed in, all pixels in calibrated image are valid) \
        to 1 (zoomed out, all pixels in original image are in calibrated image). \
            Default is 0.",
                    type=float, default=0.0)
    parser.add_argument("-s", "--show_original", help="Show original video",
                    action="store_true")
    
    args = parser.parse_args()
    
    calibrator = Calibrator()#args.model)
    calibrator.getCalibInfo(args.calibration_path)
    # calibrator.initRectifyMap(args.zoom)
    
    imgPaths = glob(args.input + "/*.png") + glob(args.input + "/*.jpg")
    outDir = args.output

    if not os.path.exists(outDir):
        os.makedirs(outDir)
    
    for imgPath in imgPaths:
        
        frame = cv2.imread(imgPath)
        
        if args.horizontal_flip:
            frame = cv2.flip(frame, 1)
        if args.vertical_flip:
            frame = cv2.flip(frame, 0)
        
        calibrator.initRectifyMap(args.zoom, (frame.shape[1], frame.shape[0]))
        
        dst = calibrator.undistort(frame)
        save = dst.copy()
        
        # Display the undistortion result
        cv2.imshow("undistorted", dst)
        if args.show_original != 0:
            cv2.imshow("original", frame)
        
        # Save the undistortion result (same name with the corresponding input image)
        name = os.path.basename(imgPath)
        cv2.imwrite("{}/{}".format(outDir, name), save)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    
import cv2
import argparse
from utils import Calibrator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", help="Input source (camera index/video path)", default='0')
    parser.add_argument("-hf", "--horizontal_flip", help="horizontal flip", action="store_true")
    parser.add_argument("-vf", "--vertical_flip", help="horizontal flip", action="store_true")

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
    
    calibrator = Calibrator()#args.fisheye)#args.model)
    calibrator.getCalibInfo(args.calibration_path)
    # calibrator.initRectifyMap(args.zoom)
    
    source = args.index
    cap = cv2.VideoCapture(eval(source) if source.isnumeric() else source)
    
    if not source.isnumeric():
        frameNum = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
    new_dim = None

    while True:
        
        # Capture frame-by-frame
        success, frame = cap.read()
        if args.horizontal_flip:
            frame = cv2.flip(frame, 1)
        if args.vertical_flip:
            frame = cv2.flip(frame, 0)
    
        if success:
            
            if new_dim is None:
                new_dim = (frame.shape[1], frame.shape[0])
                print(new_dim)
                calibrator.initRectifyMap(args.zoom, new_dim)

            # undistort
            dst = calibrator.undistort(frame)
            
            # Display the resulting frame
            cv2.imshow("undistorted", dst)
            if args.show_original != 0:
                cv2.imshow("original", frame)
        else:
            if source.isnumeric() or cap.get(cv2.CAP_PROP_POS_FRAMES) >= frameNum:
                break
            else:
                continue
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    
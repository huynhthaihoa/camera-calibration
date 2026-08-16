import cv2
import numpy as np
import os
import glob
# import argparse
import pickle
from math import sqrt, ceil, pi, acos
from enum import Enum
from scipy.optimize import least_squares

FISHEYE_CALIBRATION_FLAGS = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW#+cv2.fisheye.CALIB_CHECK_COND
PINHOLE_CALIBRATION_FLAGS = 0
RGB_CORNER_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE #| cv2.CALIB_CB_FAST_CHECK
THERMAL_CORNER_FLAG = cv2.CALIB_CB_ADAPTIVE_THRESH
SUBPIX_CRITERIA = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1
TERMINATION_CRITERIA = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6
PARAM_RANGES = [0.7, 0.7, 0.4, 0.5]
CLAHE = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8, 8))
#CLAHE = cv2.createCLAHE(clipLimit = 5.0, tileGridSize = (16, 12))#clipLimit = 5)
# CLAHE = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))#clipLimit = 5)

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def _pdist(p1, p2):
    """
    L-2 distance of two points
    
    Args:
        p1 = (x1, y1)
        p2 = (x2, y2)    
    """
    return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

def _calculate_skew(corners):
    """
    Get skew for given checkerboard detection.
    Scaled to [0,1], which 0 = no skew, 1 = high skew
    Skew is proportional to the divergence of three outside corners from 90 degrees.
    """
    # TODO Using three nearby interior corners might be more robust, outside corners occasionally
    # get mis-detected
    up_left, up_right, down_right, _ = corners

    def angle(a, b, c):
        """
        Return angle between lines ab, bc
        """
        ab = a - b
        cb = c - b
        return acos(np.dot(ab,cb) / (np.linalg.norm(ab) * np.linalg.norm(cb)))

    skew = min(1.0, 2. * abs((pi / 2.) - angle(up_left, up_right, down_right)))
    return skew

def _calculate_area(corners):
    """
    Get 2d image area of the detected checkerboard.
    The projected checkerboard is assumed to be a convex quadrilateral, and the area computed as
    |p X q|/2; see http://mathworld.wolfram.com/Quadrilateral.html.
    """
    (up_left, up_right, down_right, down_left) = corners
    a = up_right - up_left
    b = down_right - up_right
    c = down_left - down_right
    p = b + c
    q = a + b
    return abs(p[0]*q[1] - p[1]*q[0]) / 2.

def lmin(seq1, seq2):
    """ Pairwise minimum of two sequences """
    return [min(a, b) for (a, b) in zip(seq1, seq2)]

def lmax(seq1, seq2):
    """ Pairwise maximum of two sequences """
    return [max(a, b) for (a, b) in zip(seq1, seq2)]

def _dsProject(x, y, z, fx, fy, cx, cy, xi, alpha):
    """
    Double Sphere forward projection (Usenko et al., "The Double Sphere Camera
    Model", 3DV 2018): camera-frame 3D coordinates -> pixel coordinates.
    x, y, z may be scalars or numpy arrays of any (matching) shape.
    """
    d1 = np.sqrt(x**2 + y**2 + z**2)
    d2 = np.sqrt(x**2 + y**2 + (xi * d1 + z)**2)
    denom = alpha * d2 + (1. - alpha) * (xi * d1 + z)
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    u = fx * x / denom + cx
    v = fy * y / denom + cy
    return u, v

def _dsProjectPoints(objPoints, rvec, tvec, fx, fy, cx, cy, xi, alpha):
    """
    Project Nx3 object points into pixel coordinates given extrinsics (rvec,
    tvec) and Double Sphere intrinsics. Returns Nx2 image points.
    """
    R, _ = cv2.Rodrigues(rvec)
    Xc = (R @ objPoints.reshape(-1, 3).T + tvec.reshape(3, 1)).T
    u, v = _dsProject(Xc[:, 0], Xc[:, 1], Xc[:, 2], fx, fy, cx, cy, xi, alpha)
    return np.stack([u, v], axis=1)

def _dsPackParams(fx, fy, cx, cy, xi, alpha, rvecs, tvecs):
    """ Pack Double Sphere intrinsics + per-view extrinsics into a flat parameter vector """
    params = [fx, fy, cx, cy, xi, alpha]
    for rvec, tvec in zip(rvecs, tvecs):
        params.extend(np.asarray(rvec).flatten())
        params.extend(np.asarray(tvec).flatten())
    return np.array(params, dtype=np.float64)

def _dsUnpackParams(params, nViews):
    """ Inverse of _dsPackParams """
    fx, fy, cx, cy, xi, alpha = params[:6]
    rvecs, tvecs = [], []
    offset = 6
    for _ in range(nViews):
        rvecs.append(params[offset:offset + 3].reshape(3, 1))
        tvecs.append(params[offset + 3:offset + 6].reshape(3, 1))
        offset += 6
    return fx, fy, cx, cy, xi, alpha, rvecs, tvecs

def _dsResiduals(params, objPointsList, imgPointsList):
    """ Flattened reprojection residuals across all calibration views """
    nViews = len(objPointsList)
    fx, fy, cx, cy, xi, alpha, rvecs, tvecs = _dsUnpackParams(params, nViews)
    residuals = []
    for objPts, imgPts, rvec, tvec in zip(objPointsList, imgPointsList, rvecs, tvecs):
        proj = _dsProjectPoints(objPts, rvec, tvec, fx, fy, cx, cy, xi, alpha)
        obs = np.asarray(imgPts).reshape(-1, 2)
        residuals.append((proj - obs).flatten())
    return np.concatenate(residuals)

def _dsCalibrate(objPoints, imgPoints, dim):
    """
    Calibrate a Double Sphere camera from checkerboard correspondences via
    nonlinear least squares (joint optimization of intrinsics and per-view
    extrinsics), since OpenCV has no built-in solver for this model.

    Args:
        objPoints: list of per-view Nx1x3 object point arrays
        imgPoints: list of per-view Nx1x2 detected corner arrays
        dim: (width, height) of the calibration images

    Returns:
        reproj_err, K, D, rvecs, tvecs
    """
    width, height = dim
    nViews = len(objPoints)

    fx0 = fy0 = 0.5 * (width + height) / 2.
    cx0, cy0 = width / 2., height / 2.
    xi0, alpha0 = 0.0, 0.5

    K0 = np.array([[fx0, 0, cx0], [0, fy0, cy0], [0, 0, 1]], dtype=np.float64)
    rvecs0, tvecs0 = [], []
    for objPts, imgPts in zip(objPoints, imgPoints):
        _, rvec, tvec = cv2.solvePnP(np.asarray(objPts, dtype=np.float64),
                                      np.asarray(imgPts, dtype=np.float64), K0, None)
        rvecs0.append(rvec)
        tvecs0.append(tvec)

    x0 = _dsPackParams(fx0, fy0, cx0, cy0, xi0, alpha0, rvecs0, tvecs0)

    lower = [1., 1., 0., 0., -1., 1e-6] + [-np.inf] * (6 * nViews)
    upper = [np.inf, np.inf, width, height, 1., 1. - 1e-6] + [np.inf] * (6 * nViews)

    result = least_squares(_dsResiduals, x0, bounds=(lower, upper), method='trf',
                            args=(objPoints, imgPoints))

    fx, fy, cx, cy, xi, alpha, rvecs, tvecs = _dsUnpackParams(result.x, nViews)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([[xi], [alpha]], dtype=np.float64)
    reproj_err = float(np.sqrt(np.mean(result.fun**2)))
    return reproj_err, K, D, rvecs, tvecs

class CAMERA_MODEL(Enum):
    """Supported camera model

    Args:
        PINHOLE = 0
        FISHEYE = 1
        DOUBLE_SPHERE = 2
    """    
    PINHOLE = 0
    FISHEYE = 1
    DOUBLE_SPHERE = 2
    
class IMAGE_SENSOR(Enum):
    """Supported sensor

    Args:
        RGB = 0 
        THERMAL = 1
    """    
    RGB = 0
    THERMAL = 1

class CornerDetector:
    """Corner Detector class:
    
    Properties:
        img_shape: image shape
        nRows: checkerboard's number of rows
        nCols: checkerboard's number of columns
        nBorders: number of border pixels
        is_low_res: working on low resolution (True) or high resolution (False) stream 
    """    
    def __init__(self, checkerboard, is_low_res, border = 8):
        """Instance initialization

        Args:
            checkerboard (tuple): checkerboard shape
            is_low_res (bool): working on low resolution (True) or high resolution (False) stream 
            border (int): number of border pixels (default is 8)
        """        
        self.img_shape = None
        
        # Make sure n_cols > n_rows to agree with OpenCV CB detector output
        self.nCols = max(checkerboard[0], checkerboard[1])
        self.nRows = min(checkerboard[0], checkerboard[1])
        
        self.nBorders = border
        self.is_low_res = is_low_res
            
    def getOutsideCorners(self, corners):
        """Return the four corners of the board as a whole, as (up_left, up_right, down_right, down_left).

        Args:
            corners (list): input corners

        Raises:
            Exception: Invalid number of corners

        Returns:
            (up_left, up_right, down_right, down_left)
        """        
        if corners.shape[1] * corners.shape[0] != self.nCols * self.nRows:
            raise Exception("Invalid number of corners! %d corners. X: %d, Y: %d" % (corners.shape[1] * corners.shape[0],
                                                                                self.nCols, self.nRows))
        up_left    = corners[0, 0]
        up_right   = corners[self.nCols - 1, 0]
        down_right = corners[-1, 0]
        down_left  = corners[-self.nCols, 0]
        return (up_left, up_right, down_right, down_left)
    
    def getCorners(self, img, isThermal):
        """Get corners from the image

        Args:
            img: input image to get corners
            isThermal (bool): image is thermal image (True) or not (False)
        Returns:
            retval: result (True if corners are extracted successfully, False otherwise)
            corners: corner list
        """        
        if self.img_shape == None:
            self.img_shape = img.shape[:2][::-1]
        else:
            assert self.img_shape == img.shape[:2][::-1], "All images must share the same size."
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        if isThermal: # apply CLAHE on thermal image
            gray = np.array(255 - gray, dtype=np.uint8)
            gray = CLAHE.apply(gray)
            flags = THERMAL_CORNER_FLAG
        else:
            flags = RGB_CORNER_FLAGS
            
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (self.nCols, self.nRows), flags=flags)
        
        if not ret:
            ret, corners = cv2.findChessboardCornersSB(gray, (self.nCols, self.nRows), flags=cv2.CALIB_CB_EXHAUSTIVE)
        
        if not ret:
            return ret, corners
        
        # If any corners are within BORDER pixels of the screen edge, reject the detection by setting ok to false
        # NOTE: This may cause problems with very low-resolution cameras, where 8 pixels is a non-negligible fraction
        # of the image size. See http://answers.ros.org/question/3155/how-can-i-calibrate-low-resolution-cameras
        if not self.is_low_res and not all([(self.nBorders < corners[i, 0, 0] < (self.img_shape[1] - self.nBorders)) and (self.nBorders < corners[i, 0, 1] < (self.img_shape[0] - self.nBorders)) for i in range(corners.shape[0])]):
            ret = False
            
        # Ensure that all corner-arrays are going from top to bottom.
        if self.nCols != self.nRows:
            if corners[0, 0, 1] > corners[-1, 0, 1]:
                corners = np.copy(np.flipud(corners))
        else:
            direction_corners = (corners[-1] - corners[0]) >= np.array([[0.0, 0.0]])
            if not np.all(direction_corners):
                if not np.any(direction_corners):
                    corners = np.copy(np.flipud(corners))
                elif direction_corners[0][0]:
                    corners = np.rot90(corners.reshape(self.nRows, self.nCols, 2)).reshape(self.nCols * self.nRows, 1, 2)
                else:
                    corners = np.rot90(corners.reshape(self.nRows, self.nCols, 2), 3).reshape(self.nCols * self.nRows, 1, 2)
        
        if ret:
            # Use a radius of half the minimum distance between corners. This should be large enough to snap to the
            # correct corner, but not so large as to include a wrong corner in the search window.
            min_distance = float("inf")
            for row in range(self.nRows):
                for col in range(self.nCols - 1):
                    index = row * self.nRows + col
                    min_distance = min(min_distance, _pdist(corners[index, 0], corners[index + 1, 0]))
            for row in range(self.nRows - 1):
                for col in range(self.nCols):
                    index = row * self.nRows + col
                    min_distance = min(min_distance, _pdist(corners[index, 0], corners[index + self.nCols, 0]))
            radius = int(ceil(min_distance * 0.5))
            corners = cv2.cornerSubPix(gray, corners, (radius, radius), (-1, -1), SUBPIX_CRITERIA)
            
        return ret, corners

class Calibrator:
    """Calibrator class
    """    
    def __init__(self, model = 0, is_thermal = False, param_thres = 0.2, quantity_thres = 40):#, CALIBRATION_FLAGS = 0):
        """Instance initialization
        Args:
            model (int): camera model (0: pinhole, 1: fisheye, 2: double sphere)
            param_thres (float): param threshold for evaluating good detected corners
            quantity_thres (int): minimum number of good images for calibration
        """
        self.camera_model = CAMERA_MODEL(model)  # raises ValueError if model isn't 0, 1, or 2
        if self.camera_model == CAMERA_MODEL.DOUBLE_SPHERE:
            self.calibration_flags = None
        elif self.camera_model == CAMERA_MODEL.FISHEYE:
            self.calibration_flags = FISHEYE_CALIBRATION_FLAGS
        else:
            self.calibration_flags = PINHOLE_CALIBRATION_FLAGS
        if is_thermal:
            self.image_sensor = IMAGE_SENSOR.THERMAL
        else:
            self.image_sensor = IMAGE_SENSOR.RGB
        self.dim = None

        # self.imgpoints = list()
        # self.objpoints = list()
        #self.ptslen = 0
        
        # self.db is list of parameters samples for use in calibration. Parameters has form
        # (X, Y, size, skew) all normalized to [0, 1], to keep track of what sort of samples we've taken
        # and ensure enough variety.
        self.db = list()
        
        self.last_frame_corners = None
        self.good_corners = list()
        
        # Set to true when we have sufficiently varied samples to calibrate
        self.good_enough = False
        
        self.param_ranges = PARAM_RANGES
        #self._param_names = ["X", "Y", "Size", "Skew"]
        
        self.param_thres = param_thres
        
        self.quantity_thres = quantity_thres
        
        self.progress = [0., 0., 0., 0.]
    
    def setCheckerboard(self, checkerboard, is_low_res = True, border = 8):
        """Set checkerboard shape for calibration

        Args:
            checkerboard: checkerboard shape
            is_low_res: stream is low resolution (True) or not (False)
            border (int): number of border pixels (default is 8)
        """        
        self.detector = CornerDetector(checkerboard, is_low_res, border)

    def getCalibInfo(self, filename: str):
        """Get camera intrinsic parameters from file
        Args:
            filename (str): calibration file
        """        
        # with open(filename, "rb") as f:
        #     data = pickle.load(f)
        #     self.dim, self.K, self.D = data #, self.rvecs, self.tvecs = data
        #     print("DIM=" + str(self.dim))
        #     print("K=np.asarray(" + str(self.K.tolist()) + ")")
        #     print("D=np.array(" + str(self.D.tolist()) + ")")
            # print("rvecs={}\n".format(self.rvecs))
            # print("tvecs={}\n".format(self.tvecs))
        cv_file = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        
        model = cv_file.getNode("model").string()
        if model == 'pinhole':
            self.camera_model = CAMERA_MODEL.PINHOLE
        elif model == 'double_sphere':
            self.camera_model = CAMERA_MODEL.DOUBLE_SPHERE
        else:
            self.camera_model = CAMERA_MODEL.FISHEYE
        
        sensor = cv_file.getNode("sensor").string()
        if sensor == 'rgb':
            self.image_sensor = IMAGE_SENSOR.RGB
        else:
            self.image_sensor = IMAGE_SENSOR.THERMAL
        
        self.dim = tuple(cv_file.getNode("dim").mat().astype(np.uint16).transpose(1, 0).squeeze())#, self.dim)
        self.K = cv_file.getNode("K").mat()#, self.K)
        self.D = cv_file.getNode("D").mat()#, self.D)
        # cv_file.write("dim", self.dim)
        cv_file.release()
    
    # def getCalibInfo(self, dim, K, D):
    #     """Get camera intrinsic parameters from input value

    #     Args:
    #         dim: image shape 
    #         K: camera matrix
    #         D: distortion coefficients
    #     """        
    #     self.dim = dim
    #     self.K = np.asarray(K)
    #     self.D = np.asarray(D)
    
    def saveCalib(self, filename):
        """Save the calibration information to file
        Args:
            filename (str): calibration file
        """ 
        # with open(filename, "wb") as f:
        #     data = [self.dim, self.K, self.D]#, self.rvecs, self.tvecs]
        #     pickle.dump(data, f)
        cv_file = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        if self.camera_model == CAMERA_MODEL.FISHEYE:
            cv_file.write("model", "fisheye")
        elif self.camera_model == CAMERA_MODEL.DOUBLE_SPHERE:
            cv_file.write("model", "double_sphere")
        else:
            cv_file.write("model", "pinhole")
        # cv_file.write("model", str(self.camera_model))
        if self.image_sensor == IMAGE_SENSOR.RGB:
            cv_file.write("sensor", "rgb")
        else:
            cv_file.write("sensor", "ir")
        # cv_file.write("image_sensor", self.image_sensor)
        cv_file.write("dim", self.dim)
        cv_file.write("K", self.K)
        cv_file.write("D", self.D)
        cv_file.write("reproj_err", self.reproj_err)
        # cv_file.write("dim", self.dim)
        cv_file.release()
    
    def undistort(self, img):
        """Apply the post-calibration undistortion to the source image
        Args:
            img: source image

        Returns:
            undistorted image
        """        
        return cv2.remap(img, self.mapx, self.mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    def initRectifyMap(self, alpha, new_dim = None):
        """Initialize rectify map:
         alpha (float): zooming value, ranges from 0 (zoomed in, all pixels in calibrated image are valid) 
                           to 1 (zoomed out, all pixels in original image are in calibrated image).
         new_dim (int[]): (width, height) of the input image (maybe different with dim in intrinsic params)
        """        
        R = np.eye(3, dtype=np.float64)
        if new_dim is None:
            K = self.K
            new_dim = self.dim
        else:
            K = self.K * new_dim[0] / self.dim[0]
            K[2][2] = 1.0
            
        if self.camera_model == CAMERA_MODEL.PINHOLE:
            newcameramtx, _ = cv2.getOptimalNewCameraMatrix(K, self.D, new_dim, alpha)
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(K, self.D, R, newcameramtx, new_dim, cv2.CV_32FC1)
        elif self.camera_model == CAMERA_MODEL.FISHEYE:
            newcameramtx = np.zeros((3, 4), dtype=np.float64)
            newcameramtx[:3, :3] = self.K
            newcameramtx[0, 0] /= (1. + alpha)
            newcameramtx[1, 1] /= (1. + alpha)
            self.mapx, self.mapy = cv2.fisheye.initUndistortRectifyMap(K, self.D, R, newcameramtx, new_dim, cv2.CV_32FC1)
        elif self.camera_model == CAMERA_MODEL.DOUBLE_SPHERE:
            # OpenCV has no Double Sphere variant of initUndistortRectifyMap, so build the map
            # manually: for each destination (rectified) pixel, cast a ray from a virtual pinhole
            # camera and forward-project it through the Double Sphere model to find the source pixel.
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            ds_xi, ds_alpha = self.D[0, 0], self.D[1, 0]
            new_fx = fx / (1. + alpha)
            new_fy = fy / (1. + alpha)
            new_cx, new_cy = cx, cy
            u, v = np.meshgrid(np.arange(new_dim[0], dtype=np.float64), np.arange(new_dim[1], dtype=np.float64))
            x = (u - new_cx) / new_fx
            y = (v - new_cy) / new_fy
            z = np.ones_like(x)
            mapx, mapy = _dsProject(x, y, z, fx, fy, cx, cy, ds_xi, ds_alpha)
            self.mapx = mapx.astype(np.float32)
            self.mapy = mapy.astype(np.float32)

    def getParams(self, corners):
        """
        Return list of parameters [X, Y, size, skew] describing the checkerboard view:
        
        Args:
            corners: detected corners from this view
        """
        (height, width) = self.dim
        Xs = corners[:, :, 0]
        Ys = corners[:, :, 1]
        outside_corners = self.detector.getOutsideCorners(corners)
        area = _calculate_area(outside_corners)
        skew = _calculate_skew(outside_corners)
        border = sqrt(area)
        # For X and Y, we "shrink" the image all around by approx. half the board size.
        # Otherwise large boards are penalized because you can't get much X/Y variation.
        p_x = min(1.0, max(0.0, (np.mean(Xs) - border / 2) / (width  - border)))
        p_y = min(1.0, max(0.0, (np.mean(Ys) - border / 2) / (height - border)))
        p_size = sqrt(area / (width * height))
        params = [p_x, p_y, p_size, skew]
        return params
        
    def isGoodSample(self, params):#, corners, last_frame_corners):
        """
        Return true if the checkerboard detection described by params should be added to the database.
        """
        if not self.db:
            return True

        def param_distance(p1, p2):
            """Distance between 2 params (Manhattan distance)

            Args:
                p1: 1st param
                p2: 2nd param

            Returns:
                distance
            """            
            return sum([abs(a - b) for (a, b) in zip(p1, p2)])

        db_params = [sample for sample in self.db]
        d = min([param_distance(params, p) for p in db_params])
        #print "d = %.3f" % d #DEBUG
        # TODO What's a good threshold here? Should it be configurable?
        if d <= self.param_thres:
            return False

        # All tests passed, image should be good for calibration
        return True
    
    def makeObjectPoints(self, ptsNum):
        """Make list of object points

        Args:
            ptsNum (int): number of points

        Returns:
            opts: list of object points
        """        
        opts = list()
        num_pts = self.detector.nCols * self.detector.nRows # number of points in checkerboard
        for i in range(ptsNum):
            opts_loc = np.zeros((num_pts, 1, 3), np.float32)
            for j in range(num_pts):
                opts_loc[j, 0, 0] = (j // self.detector.nCols)
                opts_loc[j, 0, 1] = (j % self.detector.nCols)
                opts_loc[j, 0, 2] = 0
            opts.append(opts_loc)
        return opts
           
    def calibrateFromCorners(self, corners):
        """Calibrate camera from a list of corners
        """     
        N_OK = len(corners)
        objPoints = self.makeObjectPoints(N_OK)
        if self.camera_model == CAMERA_MODEL.PINHOLE:
            self.K = np.eye(3, dtype=np.float64)
            self.reproj_err, self.K, D, self.rvecs, self.tvecs = cv2.calibrateCamera(objPoints, corners, self.dim, self.K, 
                                                                                None, flags = self.calibration_flags)
            # OpenCV returns more than 8 coefficients (the additional ones all zeros) when CALIB_RATIONAL_MODEL is set.
            # The extra ones include e.g. thin prism coefficients, which we are not interested in.
            self.D = D.flat[:8].reshape(-1, 1)
        elif self.camera_model == CAMERA_MODEL.FISHEYE:
            intrinsics_in = np.eye(3, dtype=np.float64)
            ipts = np.asarray(corners, dtype=np.float64)
            opts = np.asarray(objPoints, dtype=np.float64)
            self.reproj_err, self.K, self.D, self.rvecs, self.tvecs = cv2.fisheye.calibrate(opts, ipts, self.dim, intrinsics_in,
                                                                                       None, flags = self.calibration_flags)
        elif self.camera_model == CAMERA_MODEL.DOUBLE_SPHERE:
            self.reproj_err, self.K, self.D, self.rvecs, self.tvecs = _dsCalibrate(objPoints, corners, self.dim)

    def detectCorners(self, img, accum=True):
        """Detect corners from image

        Args:
            img: input image
            accum: accumulate the detected corners for calibration or not (default is True)
        
        Return:
            retval: -1: no corner detected, 0: "good" corner detected, 1: "bad" corner (not good for accum) detected
            corners: detected corners  
            img: original image in case no corner detected, otherwise is the image with detected corner
            progress: calibration progress (only return when accum=True)    
        """ 
        retval = -1       
        success, corners = self.detector.getCorners(img, self.image_sensor == IMAGE_SENSOR.THERMAL)
        if success:
            if self.dim is None:
                self.dim = self.detector.img_shape
            
            # Add sample to database only if it's sufficiently different from any previous sample.
            if accum:
                params = self.getParams(corners)
                if self.isGoodSample(params):#, corners, self.last_frame_corners):
                    retval = 0
                    self.db.append(params)
                    self.good_corners.append(corners)
                    self.last_frame_corners = corners

                    if len(self.db) == 1:
                        self.min_params = params
                        self.max_params = params
                    else:
                        self.min_params = lmin(self.min_params, params)
                        self.max_params = lmax(self.max_params, params)
                    
                    # Don't reward small size or skew
                    min_params = [self.min_params[0], self.min_params[1], 0., 0.]

                    # For each parameter, judge how much progress has been made toward adequate variation
                    self.progress = [min((hi - lo) / r, 1.0) for (lo, hi, r) in zip(min_params, self.max_params, self.param_ranges)]
                    # If we have lots of samples, allow calibration even if not all parameters are green
                    # TODO Awkward that we update self.good_enough instead of returning it
                    self.good_enough = (len(self.db) >= self.quantity_thres) or all([p == 1.0 for p in self.progress])
                else:
                    retval = 1
                    success = False
            else:
                retval = 0

            cv2.drawChessboardCorners(img, (self.detector.nCols, self.detector.nRows), corners, success)
        
        if accum:
            return retval, corners, img, self.progress
        
        return retval, corners, img

    def calibrate(self):
        """Calibrate and display the result
        """        
        print("Calibrating...")
        self.calibrateFromCorners(self.good_corners)
        print("Found " + str(len(self.good_corners)) + " valid images for calibration")
        print("DIM=" + str(self.dim))
        print("K=np.asarray(" + str(self.K.tolist()) + ")")
        print("D=np.array(" + str(self.D.tolist()) + ")")
        print("rvecs={}\n".format(self.rvecs))
        print("tvecs={}\n".format(self.tvecs))
        print("reproj_err={}\n".format(self.reproj_err))
        
    def getAccumulatedLen(self):
        """Get number of accumulated samples for calibration

        Returns:
            len(self.db) (int)
        """        
        return len(self.db)
    
    def isGoodEnough(self):
        """Get good_enough value

        Returns:
           self.good_enough
        """        
        return self.good_enough
    
    def getGoodCorners(self):
        """Get good detected corners

        Returns:
           self.good_corners
        """        
        return self.good_corners
    
    def getCameraMatrix(self):
        """Get camera matrix

        Returns:
            self.K
        """        
        return self.K
    
    def getDistortionCoeffs(self):
        """Get distortion coefficients

        Returns:
            self.D
        """  
        return self.D




        


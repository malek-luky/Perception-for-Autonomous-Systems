import cv2
import numpy as np

#################################################################
################        CALIBRATED MATRICES      ################
#################################################################
K_left = np.array([[705.12855386,   0,         621.0422932 ],
[  0 ,        705.05638672, 370.57066306],
[  0  ,         0      ,     1  ,      ]])
K_right = np.array([[702.64805575  , 0     ,    649.52345869],
[  0       ,  702.90821064, 373.12894423],
[  0      ,     0       ,    1        ]])
dist_left = np.array([[-3.29479779e-01 , 1.41779399e-01 ,-1.15869227e-04 , 2.53564192e-04
-3.10092442e-02]])
dist_right = np.array([[-3.25580130e-01 , 1.39151531e-01 ,-2.55232895e-04 , 4.20204047e-04
-3.19659396e-02]])
R = np.array([[ 0.99991381, -0.00530365, -0.01201018],
[ 0.00527804 , 0.99998373, -0.00216356],
[ 0.01202145 , 0.00209999 , 0.99992553]])
T = np.array([[-1.19993826e+02],
[-2.56957545e-01],
[-5.18613288e-02]])
F = np.array([[-6.02253356e-09 ,1.09798538e-07 ,-4.24691775e-04],
[ 2.97380893e-06 , 5.39499206e-07 , 1.78876466e-01],
[-1.67147433e-03 ,-1.80656239e-01 , 1.00000000e+00]])

#################################################################
#################        MAPPING MATRICES      ##################
#################################################################
#h, w, _ = cv2.imread("conveyor_full_without/left/left-0.png").shape
h, w = 720, 1280
size = (w, h)
R_left, R_right, P_left, P_right, _, roi_left, roi_right = cv2.stereoRectify(K_left, dist_left, K_right, dist_right, (w,h), R,T, alpha=0)
leftMapX, leftMapY = cv2.initUndistortRectifyMap(K_left, dist_left, R_left, P_left, (w,h), cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(K_right, dist_right, R_right, P_right, (w,h), cv2.CV_32FC1)

def compute_depth(x, y, depth_map):
    """
    Computes the depth of the object in the camera frame.
    :param data: x, y, depth_image
    :param x: centroid of the object, x coordinate
    :param y: centroid of the object, y coordinate
    :depth_image: image depth of the conveyer belt
    :return: z (depth)
    """

    # Compute the Z coordinate
    x = int(x)
    y = int(y)
    if x>16 and y>16:
        Z = np.mean(depth_map[y:y+15,x-15:x+15])
    else:
        Z = 0
    return Z
    
def calibrate(img_left):
    """
    Calibrates the data
    :param data: frame left, frame right
    :return: calibrated_left, calibrated_right
    """

    #################################################################
    #################        RECTIFICATION       ####################
    #################################################################

    # Rectify images
    (h,w,_) = img_left.shape
    
    left_rectified = np.zeros(img_left.shape[:2], np.uint8)
    left_rectified = cv2.remap(img_left, leftMapX, leftMapY, cv2.INTER_LINEAR, left_rectified, cv2.BORDER_CONSTANT)

    return left_rectified

def get_depth_map(gray_left, gray_right):
    #################################################################
    #################         DEPTH IMAGE        ####################
    #################################################################

    # PARAMETERS
    min_disp = 0  # 22
    num_disp = 16*14  # 256
    block_size = 5  #5 
    sigma = 7 #1.5
    lmbda = 16000.0 #8000

    # DISPARITY MAP
    stereo_left = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)
    stereo_left.setMinDisparity(min_disp)
    stereo_left.setDisp12MaxDiff(1)  # 200
    stereo_left.setUniquenessRatio(1)  # 1
    stereo_left.setSpeckleRange(1)  # 10
    stereo_left.setSpeckleWindowSize(1)  # 3
    disp_left = stereo_left.compute(gray_left, gray_right)#.astype(np.float32)
    disp_left2 = cv2.normalize(disp_left, None, 255, 0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_right = stereo_right.compute(gray_right,gray_left)

    # Now create DisparityWLSFilter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_left)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    disp_filtered = wls_filter.filter(disp_left, gray_left, disparity_map_right=disp_right)
    disp_filtered[disp_filtered<-16] = -16
    disp_filtered = (disp_filtered+16)/8
    return disp_filtered
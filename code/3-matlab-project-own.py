import cv2
import glob
import numpy as np

# From matlab 
mtx = np.array([[1147.48828419389,	0,	0],
                [0.847281482146116,	1154.12643432534,	0],
                [413.839306493625,	125.934874029344,	1]]).T
            
mtx_r = np.array([[1191.78793540320,	0,	0],
                 [3.15794233942629,	1201.09936716930,	0],
                 [298.930379027692,	119.282685914931,	1]]).T

R = np.array([[0.898066037373019,	0.0213408993926283,	-0.439342643650985],
              [-0.0214251464935368,	0.999759088048663,	0.00476749009820505],
              [0.439338543283939,	0.00513145956036998,	0.898306914427317]])

T = np.array([-264.886066592313,	-1.77392898927413,	46.7689011903979])

# K [-0.404240685876510,0.679412484741657,-4.804320406255162]
# P [0.003558952648167,-1.745438223176216e-04]
# [K1 K2 P1 P2 K3]
distCoeffs1 = np.array([-0.404240685876510, 0.679412484741657,
                        0.003558952648167, -1.745438223176216e-04, -4.804320406255162])
# K [-0.426243385650803,0.630959102496925,-1.016065433982113]
# P [0.002912859612620,0.001469594828381]
distCoeffs2 = np.array([-0.426243385650803, 0.630959102496925,
                        0.002912859612620, 0.001469594828381, -1.016065433982113])
cameraMatrix1 = mtx
cameraMatrix2 = mtx_r

P1 = None # project matrix
P2 = None

rt1 = None # R|T matrix
rt2 = None

'''
# some ERROR 
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
    (640, 480), R, T)
'''


def getrtMtx(rvec,tvec):
    rmtx, _ = cv2.Rodrigues(rvec)
    return np.hstack((rmtx,tvec))

def computeProjectMtx(undistort=False):
    global P1, P2, cameraMatrix1, cameraMatrix2, rt1, rt2
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2) * 25

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    objpoints_r = []
    imgpoints_r = []


    images = glob.glob('../stereo512/left.bmp')
    images_r = glob.glob('../stereo512/right.bmp')


    for fname, fname_r in zip(images, images_r):
        
        img = cv2.imread(fname)
        img_r = cv2.imread(fname_r)

        if undistort:
            img = undistortImage(img, cameraMatrix1, distCoeffs1)
            img_r = undistortImage(img_r, cameraMatrix2, distCoeffs2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)
        #print('corners',corners)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 7), None)

        # If found, add object points, image points (after refining them)
        if ret == True and ret_r == True:
            objpoints.append(objp)
            objpoints_r.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria)
            corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1),
                                          criteria)
            imgpoints.append(corners2)
            imgpoints_r.append(corners2_r)

    ret, rotation, translation = cv2.solvePnP(objpoints[0], imgpoints[0], 
    cameraMatrix1, distCoeffs1)

    ret, rotation_r, translation_r = cv2.solvePnP(objpoints[0], imgpoints_r[0], 
    cameraMatrix2, distCoeffs2)

    rt1 = getrtMtx(rotation, translation)
    rt2 = getrtMtx(rotation_r, translation_r)

    P1 = np.dot(cameraMatrix1, rt1)
    P2 = np.dot(cameraMatrix2, rt2)

    
    l = imgpoints[0].reshape(63,2).T
    r = imgpoints_r[0].reshape(63,2).T

    print("left cam pixel\n", l.shape, l)
    print("right cam pixel\n", r.shape, r) 
    
    print("P1\n",P1, "\nP2:\n",P2)
    p4d = cv2.triangulatePoints(P1, P2, l, r)
    print("left camear p4d\n", p4d/p4d[-1])

    # check rt1 
    rtmtxl_homo = np.vstack((rt1,np.array([0,0,0,1])))
    obj_homo = cv2.convertPointsToHomogeneous(objpoints[0]).reshape(63,4).T
    print("P*RT:\n", np.dot(rtmtxl_homo, obj_homo))
    

def getImagePoints(undistort = False):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2) 

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    objpoints_r = []
    imgpoints_r = []

    images = glob.glob('../stereo512/left.bmp')
    images_r = glob.glob('../stereo512/right.bmp')

    images.sort()
    images_r.sort()

    for fname, fname_r in zip(images, images_r):
        img = cv2.imread(fname)
        img_r = cv2.imread(fname_r)

        if undistort:
            img = undistortImage(img, cameraMatrix1, distCoeffs1)
            img_r = undistortImage(img_r, cameraMatrix2, distCoeffs2)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)
        #print('corners',corners)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 7), None)

        # If found, add object points, image points (after refining them)
        if ret == True and ret_r == True:
            objpoints.append(objp)
            objpoints_r.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria)
            corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1),
                                          criteria)
            imgpoints.append(corners2)
            imgpoints_r.append(corners2_r)
    l = imgpoints[0].reshape(63, 2).T
    r = imgpoints_r[0].reshape(63, 2).T

    return l, r

def undistortImage(img,_cam_mtx, _cam_dis):
    new_image = cv2.undistort(img, _cam_mtx, _cam_dis)
    return new_image



def getp3d(imgpoints_l, imgpoints_r):
    '''
    l : left  cam imgpoints  2 * N [[x1, x2,...xn], [y1, y2,...yn]]
    r : right cam imgpoints  2 * N
    return : 3 * N  [[x1...xn], [y1...yn], [z1...zn]]
    '''
    global P1, P2
    l = imgpoints_l
    r = imgpoints_r

    p4d = cv2.triangulatePoints(P1, P2, l, r)
    X = p4d/p4d[-1]  # 3d in chessboard coor

    return X[:-1]
    

if __name__ == '__main__':
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2) * 25
    objp = objp.T

    computeProjectMtx(undistort=False)
    # check for cheeseboard 
    l, r = getImagePoints(undistort=False)
    p3d = getp3d(l, r)
    print("cheese board corners p3d:\n", p3d)
    print("MSE: ",np.sqrt(np.sum(np.square(p3d - objp)/63,axis=1)))

    # experiments  ../stereo512/left.bmp
    l = np.array([[432.0, 378.0], [244.0, 168.0]])
    r = np.array([[283.0, 207.0], [249.0, 164.0]])
    p3d = getp3d(l, r)
    print("p3d:")
    print(p3d)
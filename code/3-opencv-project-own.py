import cv2
import glob
import numpy as np


cameraMatrix1 = None
cameraMatrix2 = None

distCoeffs1 = None
distCoeffs2 = None

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

def calibration():
    # termination criteria
    global cameraMatrix1, cameraMatrix2, distCoeffs1, distCoeffs2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2) * 25

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    objpoints_r = []
    imgpoints_r = []

    images = glob.glob('../left512/*.bmp')
    images_r = glob.glob('../right512/*.bmp')
    #print(images,images_r)
    images.sort()
    images_r.sort()

    for fname, fname_r in zip(images, images_r):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_r = cv2.imread(fname_r)
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

            # Draw and display the corners
            # cv2.imshow('img', img)
            # cv2.waitKey(500)
        else:
            print("No corners found")

    ret, cameraMatrix1, distCoeffs1, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       gray.shape[::-1], None,
                                                       None)

    ret, cameraMatrix2, distCoeffs2, rvecs, tvecs = cv2.calibrateCamera(objpoints_r,
                                                           imgpoints_r,
                                                           gray_r.shape[::-1],
                                                           None, None)
    #print("mtx mtx_r\n",cameraMatrix1, "\n", cameraMatrix2)
    #print("dis\n", distCoeffs1, "\n", distCoeffs2)
    #print("Intrinsic Done.")

def getrtMtx(rvec,tvec):
    rmtx, _ = cv2.Rodrigues(rvec)
    return np.hstack((rmtx,tvec))

def computeProjectMtx(undistort=False, pointnumber=4):
    global P1, P2, cameraMatrix1, cameraMatrix2, rt1, rt2, distCoeffs1, distCoeffs2

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2) * 25

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

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
    # random choose some points
    
    randn_index = np.random.randint(0,63,pointnumber)

    objpoints_ = [objpoints[0][randn_index]]
    imgpoints_ = [imgpoints[0][randn_index]]
    imgpoints_r_ = [imgpoints_r[0][randn_index]]
    ret, rotation, translation = cv2.solvePnP(objpoints_[0], imgpoints_[0], 
    cameraMatrix1, distCoeffs1)

    ret, rotation_r, translation_r = cv2.solvePnP(objpoints_[0], imgpoints_r_[0], 
    cameraMatrix2, distCoeffs2)

    rt1 = getrtMtx(rotation, translation)
    rt2 = getrtMtx(rotation_r, translation_r)

    P1 = np.dot(cameraMatrix1, rt1)
    P2 = np.dot(cameraMatrix2, rt2)

    
    l = imgpoints[0].reshape(63,2).T
    r = imgpoints_r[0].reshape(63,2).T

    #print("left cam pixel\n", l.shape, l)
    #print("right cam pixel\n", r.shape, r) 
    
    #print("P1\n",P1, "\nP2:\n",P2)
    p4d = cv2.triangulatePoints(P1, P2, l, r)
    #print("left camear p4d\n", p4d/p4d[-1])

    # check rt1 
    rtmtxl_homo = np.vstack((rt1,np.array([0,0,0,1])))
    obj_homo = cv2.convertPointsToHomogeneous(objpoints[0]).reshape(63,4).T
    #print("P*RT:\n", np.dot(rtmtxl_homo, obj_homo))
    

def getImagePoints(undistort = False):
    global P1, P2, cameraMatrix1, cameraMatrix2, rt1, rt2, distCoeffs1, distCoeffs2
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
    

import matplotlib.pyplot as plt  
def plot_mse(pn,_x,_y,_z):

    l1=plt.plot(pn,_x,'r--',label='X')
    l2=plt.plot(pn,_y,'g--',label='Y')
    l3=plt.plot(pn,_z,'b--',label='Z')
    plt.plot(pn,_x,'ro-',pn,_y,'g+-',pn,_z,'b^-')
    plt.title('MSE-Different Point Number for RT')
    plt.xlabel('The point number compute RT')
    plt.ylabel('The MSE')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2) * 25
    objp = objp.T
 
    calibration()
    computeProjectMtx(undistort = False,pointnumber=63)
    # check for cheeseboard 
    l, r = getImagePoints(undistort=False)
    p3d = getp3d(l, r)

    print("cheese board corners p3d:\n", p3d)
    print("MSE: ",np.sqrt(np.sum(np.square(p3d - objp)/63, axis = 1)))
    print("MSE2: ",np.sqrt(np.sum(np.square(p3d[-1]-0)/63)))


    # experiments  ../stereo512/left.bmp
    l = np.array([[432.0, 378.0], [244.0, 168.0]])
    r = np.array([[283.0, 207.0], [249.0, 164.0]])
    p3d = getp3d(l, r)
    print("p3d:")
    print(p3d)

    # RT 矩阵与点数的关系
    l, r = getImagePoints(undistort=True)
    pn = [(i+4) for i in range(10)]
    _x, _y, _z= [], [], []
    loop = 100
    for _ in pn:
        tmp = np.array([0., 0., 0.])
        for i in range(loop):
            computeProjectMtx(undistort = True, pointnumber=_)
            p3d = getp3d(l, r)
            #print("cheese board corners p3d:\n", p3d)
            tmp += np.sqrt(np.sum(np.square(p3d-objp)/63, axis=1))
        tmp = list(tmp/loop)
        _x.append(tmp[0])
        _y.append(tmp[1])
        _z.append(tmp[2])
    plot_mse(pn,_x,_y,_z)
    

    


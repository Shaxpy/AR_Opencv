# Script for Camera Calibration using 7x6 Chessboard
# images - ../opencv/samples/data/left01-left14.jpg 
import numpy as np 
import cv2 as cv
import glob

# Criteria of termination
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Chessboard size define
row = 6
col = 7

# Getting Object points
objp = np.zeros((row * col, 3), np.float32)
objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2)

objp = objp.reshape(-1, 1, 3)

# Defining Arrays to be used later for storing object points and image points
objpoints = []  # 3d -> real world space
imgpoints = []  # 2d -> image plane.
images = glob.glob('imga/*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Now we have corner points
        imgpoints.append(corners)


# Printing the object points
print('Objects points: {0}, image points: {1}'.format(objpoints, imgpoints))
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)
img2 = cv.imread('imga/left11.jpg')
h,  w = img2.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort using remapping, mapping from distorted->undistorted img
mapx, mapy = cv.initUndistortRectifyMap(
    mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)
cv.imshow('img', img2)
cv.waitKey(500)

# Re-projection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error/len(objpoints)))
cv.destroyAllWindows()

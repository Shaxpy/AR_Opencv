# Script-> 3D Augmented Reality

import cv2
import numpy as np
import math
from obj_module import *
import sys
import aruco_module as aruco
from global_constants import *
from utils import get_extended_RT

if __name__ == '__main__':
    obj = find_3d_object('data/3D_assets/tes/tes.obj',
                         'data/3D_assets/tes/19.png')
    # Loading cybertruck obj

    marker_colored = cv2.imread('data/aruco.png')
    assert marker_colored is not None, "aruco marker not found"
    # Lateral inversion of camera
    marker_colored = cv2.flip(marker_colored, 1)

    marker_colored = cv2.resize(
        marker_colored, (480, 480), interpolation=cv2.INTER_CUBIC)
    marker = cv2.cvtColor(marker_colored, cv2.COLOR_BGR2GRAY)

    print("trying to access the camera")
    cv2.namedWindow("camera")
    vc = cv2.VideoCapture(0)
    assert vc.isOpened(), "Camera inaccessible"

    h, w = marker.shape
    # considering all 4 rotations
    marker_sig1 = aruco.get_bit_sig(marker, np.array(
        [[0, 0], [0, w], [h, w], [h, 0]]).reshape(4, 1, 2))
    marker_sig2 = aruco.get_bit_sig(marker, np.array(
        [[0, w], [h, w], [h, 0], [0, 0]]).reshape(4, 1, 2))
    marker_sig3 = aruco.get_bit_sig(marker, np.array(
        [[h, w], [h, 0], [0, 0], [0, w]]).reshape(4, 1, 2))
    marker_sig4 = aruco.get_bit_sig(marker, np.array(
        [[h, 0], [0, 0], [0, w], [h, w]]).reshape(4, 1, 2))

    sigs = [marker_sig1, marker_sig2, marker_sig3, marker_sig4]

    rval, frame = vc.read()
    assert rval, "Camera inaccessible"
    h2, w2,  _ = frame.shape

    h_canvas = max(h, h2)
    w_canvas = w + w2

    while rval:
        rval, frame = vc.read()  # fetch frame from camera
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):  # Escape key to exit the program
            break

        canvas = np.zeros((h_canvas, w_canvas, 3), np.uint8)  # final display
        canvas[:h, :w, :] = marker_colored  # marker for reference

        success, H = aruco.find_homography_aruco(frame, marker, sigs)
        # success = False
        if not success:
            # print('homograpy est failed')
            canvas[:h2, w:, :] = np.flip(frame, axis=1)
            cv2.imshow("3D AR", canvas)
            continue

        R_T = get_extended_RT(A, H)
        transformation = A.dot(R_T)

        # flipped for better control
        augmented = np.flip(
            augment(frame, obj, transformation, marker), axis=1)
        canvas[:h2, w:, :] = augmented
        cv2.imshow("3D AR", canvas)

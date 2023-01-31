# Script -> 2D Augmented Reality using Opencv

import cv2
import numpy as np

minimum_match = 20
# Starting an ORB instance
ORB_detect = cv2.ORB_create(nfeatures=8000)

# FLANN MATCHER algo
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params, search_params)


def load_input():
    input_image = cv2.imread('elon.jpeg')
    augment_image = cv2.imread('spaceex.jpg')

    input_image = cv2.resize(input_image, (300, 400),
                             interpolation=cv2.INTER_AREA)
    augment_image = cv2.resize(augment_image, (300, 400))
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # Keypoints with ORB
    keypoints, descriptors = ORB_detect.detectAndCompute(gray_image, None)

    return gray_image, augment_image, keypoints, descriptors


def compute_matches(descriptors_input, descriptors_output):
    # Match descriptors
    if((descriptors_output) is not None and (descriptors_input) is not None):
        matches = flann.knnMatch(np.asarray(descriptors_input, np.float32), np.asarray(
            descriptors_output, np.float32), k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.69*n.distance:
                good.append(m)
        return good
    else:
        return None


if __name__ == '__main__':

    # Getting Information form the Input image
    input_image, aug_image, input_keypoints, input_descriptors = load_input()

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    while(ret):
        ret, frame = cap.read()
        if(len(input_keypoints) < minimum_match):
            continue
        frame = cv2.resize(frame, (600, 450))
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        # ORB for keypoint detection
        output_keypoints, output_descriptors = ORB_detect.detectAndCompute(
            frame_bw, None)
        matches = compute_matches(input_descriptors, output_descriptors) 
        # Keypoint matching 
        if(matches != None):
            if(len(matches) > 20):
                src_pts = np.float32(
                    [input_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [output_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                # building Homography matrix
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                pts = np.float32([[0, 0], [0, 399], [299, 399], [
                                 299, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                M_aug = cv2.warpPerspective(aug_image, M, (600, 450))

                # Addition of Masking Image
                frameb = cv2.fillConvexPoly(frame, dst.astype(int), 0)
                Final = frameb+M_aug

                cv2.imshow('Final Output', Final)

            else:
                cv2.imshow('Final Output', frame)
        else:
            cv2.imshow('Final Output', frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):  # Escape key to exit the program
            break

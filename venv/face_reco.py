import numpy as np
import cv2

frame_number = 0

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6 * 9, 3), np.float32)  # (6,9) = chessboard size
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objpoints = []
imgpoints = []

cv2.namedWindow("video preview")
vc = cv2.VideoCapture(0)  # or cv2.VideoCapture('sourcevideo.avi')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1080, 720))

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:

    # CORNERS DETECTION

    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(frame, (9, 6), None)

    if ret == True:

        frame_number += 1
        print(frame_number)

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(frame, (9, 6), corners, ret)
        print(ret)

        # CAMERA CALIBRATION

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("ret:", ret)

        h, w = gray.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(gray, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        print("total error: ", mean_error / len(objpoints))

        out.write(frame)

    key = cv2.waitKey(20)
    if key == 27:  # Esc
        break

vc.release()
out.release()
cv2.destroyWindow("preview")

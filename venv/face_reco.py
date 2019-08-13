import numpy as np
import cv2

cv2.namedWindow("video preview")
vc = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(frame, (9, 6), None)
    cv2.drawChessboardCorners(frame, (9, 6), corners, ret)
    print(ret)
    #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    #np.savez("pose/webcam_calibration_ouput", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    out.write(frame)

    key = cv2.waitKey(20)
    if key == 27:
        break

vc.release()
out.release()
cv2.destroyWindow("preview")

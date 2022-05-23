from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time


def track_ball(frame, hsv, lower, upper, color, pts):
    # RED MASK
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    x_out = None
    y_out = None

    #  DRAW THE CENTER + CIRCLES
    if len(cnts) > 0:
        # find the largest contour in the mask
        # use it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x_out, y_out), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv2.circle(frame, (int(x_out), int(y_out)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, color, -1)
    pts.appendleft(center)

    # DRAW THE LINE
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        cv2.line(frame, pts[i - 1], pts[i], color, 2)

    return [frame, x_out, y_out, pts]


# VARIABLE INIT
blue_lower = (90, 70, 70)
blue_upper = (100, 255, 255)
blue_color = (255, 0, 0)  # bgr
pts_blue = deque(maxlen=20)

red_lower = (0, 180, 150)
red_upper = (9, 255, 255)
red_color = (0, 0, 255)
pts_red = deque(maxlen=20)

yellow_lower = (20, 100, 100)
yellow_upper = (40, 255, 255)
yellow_color = (0, 255, 255)
pts_yellow = deque(maxlen=20)

cap = cv2.VideoCapture("rolling_ball_challenge.mp4")
cap.set(3, 1280)  # width
cap.set(4, 720)  # height

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # YELLOW
    [frame_res, x_yellow, y_yellow, pts_yellow] = track_ball(
        frame, hsv, yellow_lower, yellow_upper, yellow_color, pts_yellow
    )
    print("YELLOW", x_yellow, y_yellow)

    # RED
    [frame_res, x_red, y_red, pts_red] = track_ball(
        frame, hsv, red_lower, red_upper, red_color, pts_red
    )
    print("RED", x_red, y_red)

    # BLUE
    [frame_res, x_blue, y_blue, pts_blue] = track_ball(
        frame, hsv, blue_lower, blue_upper, blue_color, pts_blue
    )
    print("BLUE", x_blue, y_blue)

    # SHOW THE IMAGES
    cv2.imshow("Frame", frame_res)
    key = cv2.waitKey(1) & 0xFF

    # PROGRAM TERMINATION
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

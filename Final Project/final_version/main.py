import glob
from functions import *
import cv2
from cv2 import bitwise_and
import numpy as np
from functions.kalman_tracker import Kalman_tracker
from tensorflow import keras
from functions.calib import *

import matplotlib.pyplot as plt

if __name__ == "__main__":
    ###################### for proper jupyter imports:
    #!pip install import-ipynb #install this before first run
    # import import_ipynb
    #note for myself: run with Python3.10 (not 64bit)
    #use the pythonian way:
    # from calibration import calibrate
    # from depth_map import compute_depth
    # from tracking import track
    # from neural_net import initialize_network, classify
    # or run with the following:
    #################################################

    # INITIALIZATION
    cnn_model = keras.models.load_model("cnn_model_3")
    tracker = Kalman_tracker(occlusion=True, Verbose=False, also_predict=True, model=cnn_model, eligibility_trace=0.75)
    
    # IMPORT IMAGES
    path_left = "with_occlusion/left/*.png" #without_occlusion
    path_right = "with_occlusion/left/*.png" #without_occlusion
    images_left = glob.glob(path_left)
    assert images_left, "No images found in {}".format(path_left)
    images_right = glob.glob(path_right)
    assert images_right, "No images found in {}".format(path_right)

    # CREATE VIDEO
    create_output_video = False
    if create_output_video:
        img_buffer = []
        FPS = 30 
        out = cv2.VideoWriter('final_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), FPS, size)

    # DEPTH MAP
    rectified_left = calibrate(cv2.imread("without_occlusion/left/1585434279_805531979_Left.png"))
    rectified_right = calibrate(cv2.imread("without_occlusion/right/1585434279_805531979_Right.png"))
    depth_map = get_depth_map(cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY))

    # MAIN
    no_of_frames = len(images_left)
    for i in range(no_of_frames):
        frame_left = cv2.imread(images_left[i])
        frame_right = cv2.imread(images_right[i])

        rectified_left = calibrate(frame_left)
        track_frame, filtered, measured, prediction = tracker.next_frame(rectified_left)
        centroidx, centroidy = filtered[0], filtered[1]
        z = compute_depth(centroidx, centroidy, depth_map)


        # ADD TEXT
        cv2.putText(track_frame, "________________",
            (700, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(track_frame, "green = measured   red = filtered",
            (700, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(track_frame, "pixel X: {}".format(str(np.int32(centroidx))),
            (700, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(track_frame, "pixel Y: {}".format(str(np.int32(centroidy))),
            (700, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if z==0: # no depth
            cv2.putText(track_frame, "depth Z: {}".format("not available"),
                        (700, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else: # Depth
            cv2.putText(track_frame, "depth Z: {}".format(str(np.round(120-z,2))),
                        (700, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)



        if create_output_video:
            img_buffer.append(track_frame)
        
        #Add more text to frame
        cv2.imshow("Perception project", track_frame) #roi

        if cv2.waitKey(1) & 0xFF == ord('q'): #stop on q
                break

        #print("\rFrame {}/{}".format(i+1, no_of_frames+1), end="")

    # create_video(buffer)
    cv2.destroyAllWindows()
    if create_output_video:
        for i in range(len(img_buffer)):
            out.write(img_buffer[i])
        out.release()
        print("video created")






"""
The Tracker, extended with a online kalman filter.

This is the interface:
    - roi, centroidx, centroidy, tracked_object   = track(calibrated_left, i)
"""

from tabnanny import verbose
import numpy as np
import cv2
from torch import float64
from functions.tracker import EuclideanDistTracker

class Kalman_tracker:

    def __init__(self, occlusion=True, Verbose=False, also_predict=False, model=None, eligibility_trace=0.95) -> None:
        
        self.occlusion = occlusion
        self.verbose=Verbose
        self.object_detector = cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=1200)
        self.tracker = EuclideanDistTracker()
        self.also_predict = also_predict
        if self.also_predict:
            self.e_trace = eligibility_trace
            self.pred = ["cup", "box", "book"]
            if model is None:
                self.cnn_model = keras.models.load_model('cnn_model_3')
            else:
                self.cnn_model = model
            self.reset_predictions()


        ### Initialize Kalman filter ###
        # The initial state (4x1).
        self.states = np.array([[0],  # Pos along the x-axis
                                [0]])  # Pos along the y-axis 

        # The initial uncertainty (2x2) - We start with some very large values.
        # Then we will fully trust measurements first iteration.
        self.P = np.array([[1000, 0],
                    [0, 1000]])

        # The external motion (2x1).
        self.u = np.array([[-4.420334045148261], # Measured average xvel
                    [1.2415468250308124]]) # Measured average yvel

        # The transition matrix (2x2). 
        self.F = np.array([[1, 0],
                    [0, 1]])

        # The observation matrix (2x4).
        self.H = np.array([[1, 0],
                    [0, 1]])

        # The measurement uncertainty.
        # How little do you want to trust the measurements?
        self.R = 75*np.array([[1, 0],
                        [0, 1]])

        self.bad_R = 100*self.R.copy()

        # Disturbance matrix
        # How little do you want to trust the model?
        self.Q = 1*np.array([[1, 0],
                        [0, 1]])

    def reset_predictions(self):
        """ Reset predictions """
        self.prediction = np.array([0, 0, 0], dtype=np.float32)
        self.prev_prediction = np.array([0, 0, 0], dtype=np.float32)
    
    @staticmethod
    def update(x, P, Z, H, R):
        ### Insert update function
        y = Z - H@x
        s = H@P@H.T + R
        K = P@H.T@np.linalg.pinv(s)
        new_x = x + K@y
        new_P = (np.identity((K@H).shape[0]) - K@H)@P
        return new_x, new_P
    
    @staticmethod
    def predict(x, P, F, u, Q):
        ### insert predict function
        new_x = F@x + u
        new_P = F@P@F.T + Q # Q=0. No internal disturbtion
        return new_x, new_P

    def next_frame(self, clean_frame):
        """ Tracks one frame, crops the objective """
        frame = clean_frame.copy()
        # Current measurements
        z = np.array([[None],  # xpos
                      [None]]) # ypos
        ncrop_img = None

        # 1
        # Object detection
        mask = self.object_detector.apply(frame) # Add new frame
        _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(mask, None, iterations=6)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 2
        # Only keep biggest objects detected/"most moving"
        biggest_area=0
        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area>biggest_area:
                biggest_area=area
                best_cnt = cnt
        # 3 
        # Constrain/limit search using heuristics
        if (biggest_area>4500 and biggest_area<76000): # reasonable size
            x, y, w, h = cv2.boundingRect(best_cnt)
            box_id = self.tracker.update([[x, y, w, h]])[0] # Track object
            
            if (x>200 and x<1200)and(y>200 and y<550): # On the conveyor
                x, y, w, h, id = box_id
                # Increase box bound to better match objects.
                rx=x-25
                ry=y-25
                rw=w+50
                rh=h+50
                centerx=(rx+rx+rw)//2
                centery=(ry+ry+rh)//2
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 3)
                cv2.circle(frame,(centerx,centery),5,(0,255,0), -1)
                
                # Cropping for classification
                if self.also_predict:
                    crop_img = frame[y:y+h, x:x+w].copy()
                    ncrop_img = cv2.resize(crop_img, (128, 128), interpolation = cv2.INTER_AREA) # Ready for classification

                    gray_object = np.array([
                                        cv2.cvtColor(ncrop_img, cv2.COLOR_RGB2GRAY).reshape((128, 128,1))
                                        ])
                    if self.verbose:
                        cv2.imshow("Object", ncrop_img)
                    #print(gray_object.shape)
                    self.prediction = np.array(self.cnn_model.predict(gray_object)[0])

                # Kalman update from measurements
                z = np.array([[centerx],  # xpos
                              [centery]]) # ypos

                # Occlusion limits
                if self.occlusion and (centerx>550 and centerx<1100): # Close to occlusion
                    self.states, self.P = self.update(self.states, self.P, z, self.H, self.bad_R)
                    if self.verbose:
                        cv2.putText(frame, "Measurement trust: {}".format('Low'), (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else: # Far from occlusion
                    self.states, self.P = self.update(self.states, self.P, z, self.H, self.R)
                    if self.verbose:
                        cv2.putText(frame, "Measurement trust: {}".format('High'), (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            else: # Off the conveyor
                # Reset model uncertainty
                self.P = np.array([[1000, 0],
                                   [0, 1000]])
                # Reset predictions
                cv2.putText(frame, "PREDICTION: {}".format("None"),
                    (700, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
                self.reset_predictions() 

        else: # Unreasonable object
            pass

        # Kalman prediction 
        self.states, self.P = self.predict(self.states, self.P, self.F, self.u, self.Q)
        cv2.circle(frame, (np.int32(self.states[0, 0]), np.int32(self.states[1, 0])), 8, color=(0, 0, 255), thickness=-1)

        # Eligibility decay
        self.prediction += self.prev_prediction
        self.prev_prediction = self.prediction.copy() * self.e_trace

        if np.sum(self.prediction)>0.05: # Some certainty
            cv2.putText(frame, "PREDICTION: {}".format(self.pred[np.argmax(self.prediction)]),
             (700, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)

        return frame, self.states.reshape((2,)), z.reshape((2,)), self.prediction


if __name__ == "__main__":
    """
    Some Auxilliary testing.
    """

    import glob
    from tensorflow import keras
    print("\n\n\n\nLocal test\n\n\n\n")
    #####################################################################################################
    #images = glob.glob("video/sample/left/*.png")

    # Distorted
    #images = glob.glob("video/full/without_occlusions/left/*.png")
    #images = glob.glob("video/full/with_occlusions/left/*.png")

    # Undistorted
    #images = glob.glob("video/full_undistorted/without_occlusions/*.png")
    #images = glob.glob("video/full_undistorted/with_occlusions/*.png")

    # Unrect - if this works its the best!
    #images = glob.glob("C:/Users/henri/OneDrive/Desktop/DTU courses/31392 Perception/final_project/tracking/video/full_unrect/without_occlusions/*.png")
    images = glob.glob("C:/Users/henri/OneDrive/Desktop/DTU courses/31392 Perception/final_project/tracking/video/full_unrect/with_occlusions/*.png")

    #####################################################################################################
    # Object detection
    no_of_frames = len(images)-1
    print("Frames:", no_of_frames)

    # CNN classification
    cnn_model = keras.models.load_model('C:/Users/henri/OneDrive/Desktop/DTU courses/31392 Perception/final_project/FULL_PROJECT/cnn_model_3');
    preds = ["cup", "box", "book"]
    previous_prediction = np.array([0, 0, 0])
    prediction = np.array([0, 0, 0])

    # Init tracker
    tracker = Kalman_tracker(occlusion=True, Verbose=False, also_predict=True, model=cnn_model, eligibility_trace=0.75)
    # The movie/image loop
    for i in range(no_of_frames):
        frame = cv2.imread(images[i]) 
        track_frame, filtered, measured, prediction = tracker.next_frame(frame)

        cv2.imshow("Perception project", track_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #stop on q
                break

        
        
    cv2.destroyAllWindows()

    

#from types import NoneType
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def update(x, P, Z, H, R):
    ### Insert update function
    y = Z - H@x
    s = H@P@H.T + R
    K = P@H.T@np.linalg.pinv(s)
    new_x = x + K@y
    new_P = (np.identity((K@H).shape[0]) - K@H)@P
    return new_x, new_P
    
def predict(x, P, F, u, Q):
    ### insert predict function
    new_x = F@x + u
    new_P = F@P@F.T + Q # Q=0. No internal disturbtion
    return new_x, new_P

# For plotting Gaussians: certainty of measurements (P-matrix)
def f(u, sigma2, x):
    return 1/np.sqrt(2*np.pi*sigma2) * np.exp(-0.5* ((x-u)**2/sigma2))
    
def offline_kalman(measurements, none_window=10):
    """
    Runs the kalman filter in offline mode.
        - measurements expected to be a 2xN numpy array.
    
    Returns the filtered x-values as a 4xN numpy array.
    """

    ### Initialize Kalman filter ###
    # The initial state (4x1).
    x = np.array([[0],# Position along the x-axis
                [0],  # Velocity along the x-axis 
                [0],  # Position along the y-axis
                [0]]) # Velocity along the y-axis 


    # The initial uncertainty (4x4) - We start with some very large values.
    # Then we will fully trust measurements first iteration.
    P = np.array([[1000, 0, 0, 0],
                  [0, 1000, 0, 0],
                  [0, 0, 1000, 0],
                  [0, 0, 0, 1000]])

    # The external motion (4x1).
    u = np.array([[0],
                  [0],
                  [0],
                  [0]]) 

    # The transition matrix (4x4). 
    F = np.array([[1, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 1],
                  [0, 0, 0, 1]])

    # The observation matrix (2x4).
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])

    # The measurement uncertainty.
    # How little do you want to trust the measurements?
    R = 1*np.array([[50, 0],
                    [0, 50]])

    # Disturbance matrix
    # How little do you want to trust the model?
    Q = 1*np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # State history
    xs = np.zeros((4, len(measurements[0])))
    for i in range(len(measurements[0])):
        new_measurement = True

        if measurements[0, i] is None:
            new_measurement = False

        if new_measurement:
            z = np.array([[measurements[0, i]], # xpos
                        [measurements[1, i]]])  # ypos


            # Update based on new measurements + previous prediction
            x, P = update(x, P, z, H, R)

        # Predict based on model
        x, P = predict(x, P, F, u, Q)
        xs[:, i] = x.reshape((4,))

        #################################################
        ### Speed is const - so we can average it out ###
        #################################################         
        # Worst fit - fitted on occlusion data.
        #x[1] = -2.6146684471289876
        #x[3] = 1.5139821443624342 
        
        # Best fit! Fitted on non-occlusion.
        #x[1] = -3.7072721104332675
        #x[3] = 1.2003268965304754

        #if i>100: # Sliding mean
        #    #Using the mean
        #    x[1] = stats.trim_mean(xs[1, -90:i], 0.1)
        #    x[3] = stats.trim_mean(xs[3, -10:i], 0.1)
            
        
    return xs
import pandas as pd
import numpy as np

def predict(X,theta):
    return X.dot(theta)



def compute_cost(X,y,theta):
    m = len(y)
    predictions = predict(X,theta)
    cost = (1/(2 * m)) * np.sum((predictions - y)**2)
    return cost



def gradient_descent(X,y,theta,learning_rate=0.01, iterations=1000):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        predictions = predict(X, theta)
        error = predictions - y
        
        theta -= (1/m) * learning_rate * (X.T.dot(error))
        cost_history[i] = compute_cost(X, y, theta)
    return theta , cost_history
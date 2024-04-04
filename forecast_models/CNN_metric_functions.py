import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


def KGE(y_true, y_pred):
    
    eps = 1e-10
    
    correlation = tfp.stats.correlation(y_true, y_pred)
    sigmaF = np.nanstd(y_pred) + eps
    sigmaT = np.nanstd(y_true) + eps
    meanF = np.nanmean(y_pred) + eps
    meanT = np.nanmean(y_true) + eps

    score = 1 - np.sqrt((correlation - 1) ** 2 + ((sigmaF / sigmaT) - 1) ** 2 + ((meanF / meanT) - 1) ** 2)
    return score

# Define the KGE metric as a TensorFlow metric
def kge_metric(y_true, y_pred):
    score = tf.py_function(KGE, [y_true, y_pred], tf.float32)
    return score

def taylor_score(y_true, y_pred):
    
    eps = 1e-10
    
    forecast_variance = np.nanstd(y_pred)/np.nanstd(y_true) + eps
    correlation = tfp.stats.correlation(y_true, y_pred)
    max_correlation = 1.0
    
    score = 4*(1+correlation) /( (forecast_variance + (1/forecast_variance) )**2 * ( 1 + max_correlation ) )
    return score

def taylor_score_metric(y_true, y_pred):
    score = tf.py_function(taylor_score, [y_true, y_pred], tf.float32)
    return score

def correlation_score(y_true, y_pred):
    
    eps = 1e-10
    
    correlation = tfp.stats.correlation(y_true + eps, y_pred + eps)
    
    return correlation

def correlation_metric(y_true, y_pred):
    score = tf.py_function(correlation_score, [y_true, y_pred], tf.float32)
    return score



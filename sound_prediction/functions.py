import wget
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import Sequential,Model
from keras.layers import concatenate,Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from sklearn.metrics import roc_curve
from keras.utils import np_utils
import numpy as np
import librosa
import librosa.display
import pylab
import os
import cv2
def feature_extractor(row):
    name = row[0]

    audio, sr = librosa.load('sound_prediction/files/temp.wav', mono=True, )
    print(audio)
    print(sr)
    # For MFCCS
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=39)
    mfccsscaled = np.mean(mfccs.T, axis=0)
    print(mfccsscaled)

    # Mel Spectogram
    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    melspec = librosa.feature.melspectrogram(y=audio, sr=sr)
    s_db = librosa.power_to_db(melspec, ref=np.max)
    librosa.display.specshow(s_db)

    savepath = os.path.join('', name + '.png')
    pylab.savefig(savepath, bbox_inches=None, pad_inches=0)
    pylab.close()

    return mfccsscaled, savepath


features = []
diagnoses = []
imgpaths = []

# for row in (data_):
#   mfccs,savepath  = feature_extractor(row)
#   print(mfccs)
#   features.append(mfccs)
#   imgpaths.append(savepath)
#   diagnoses.append([row[3],row[4]])


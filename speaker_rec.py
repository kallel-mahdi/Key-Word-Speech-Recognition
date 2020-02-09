import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from featureextraction import extract_features
import warnings


source = "development_set/"
modelpath = "speaker_models/"
test_file = "development_set_test.txt"
file_paths = open(test_file, 'r')

gmm_files = [os.path.join(modelpath, fname) for fname in
             os.listdir(modelpath)]

models = [cPickle.load(open(fname, 'rb')) for fname in gmm_files]

speakers = [fname.split("/")[-1].split("_gmm")[0] for fname
            in gmm_files]
error = 0
total_sample = 0.0

def who_speaks(aud):
    sr, audio = read(aud)
    vector = extract_features(audio, sr)

    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    return(speakers[winner])


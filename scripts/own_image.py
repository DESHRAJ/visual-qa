import argparse
import random
from PIL import Image
import subprocess
from os import listdir
from os.path import isfile, join
import os
from keras.models import model_from_json

from spacy.en import English
import numpy as np
import scipy.io
from sklearn.externals import joblib

from features import get_questions_tensor_timeseries, get_images_matrix


MODEL_PATH = '../models/lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3.json'
WEIGHT_PATH = '../models/lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3_epoch_070.hdf5'
CAFFE_MODEL_PATH = 'VGG_ILSVRC_16_layers.caffemodel'
CAFFE_PATH = 'vgg_features.prototxt'

nlp = English()
labelencoder = joblib.load('../models/labelencoder.pkl')

model = model_from_json(open(MODEL_PATH).read())
model.load_weights(WEIGHT_PATH)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

def predict():
    path = str(raw_input('Enter path to image : '))
    question = unicode(raw_input("Ask a question: "))
    print(question, path)
    if path != 'same':
        base_dir = os.path.dirname(path)
        os.system('python extract_features.py --caffe ' + str(CAFFE_PATH) + ' --model_def vgg_features.prototxt --gpu --model ' + str(CAFFE_MODEL_PATH) + ' --image ' + path )

    print 'Loading VGGfeats'
    vgg_model_path = os.path.join(base_dir + '/vgg_feats.mat')
    features_struct = scipy.io.loadmat(vgg_model_path)
    VGGfeatures = features_struct['feats']
    print "Loaded"

    timesteps = len(nlp(question))
    X_q = get_questions_tensor_timeseries([question], nlp, timesteps)
    X_i = np.reshape(VGGfeatures, (1, 4096))

    X = [X_q, X_i]

    y_predict = model.predict_classes(X, verbose=0)
    ans = labelencoder.inverse_transform(y_predict)
    print(ans)
    return 'OK'

app.run()

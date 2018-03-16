# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
from tqdm import tqdm
from keras import optimizers

from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score
import time

x_train_f= []
x_test_f= []
y_train_f= []

x_train_w= []
x_test_w= []
y_train_w= []


df_train_f = pd.read_csv('train_v1.csv')
df_test_f = pd.read_csv('sample_submission_v2.csv')
df_train_w = pd.read_csv('train_v1.csv')
df_test_w = pd.read_csv('sample_submission_v2.csv')
data_dir_haze='../1/{}.jpg'
data_dir_nhaze='../1/{}.jpg'

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train_f['tags'].values])))

labels = ['blow_down',
 'bare_ground',
 'conventional_mine',
 'blooming',
 'cultivation',
 'artisinal_mine',
 'haze',
 'primary',
 'slash_burn',
 'habitation',
 'clear',
 'road',
 'selective_logging',
 'partly_cloudy',
 'agriculture',
 'water',
 'cloudy']

f_label_map = {'agriculture': 14,
 'artisinal_mine': 5,
 'bare_ground': 1,
 'blooming': 3,
 'blow_down': 0,
 'conventional_mine': 2,
 'cultivation': 4,
 'habitation': 9,
 'primary': 7,
 'road': 11,
 'selective_logging': 12,
 'slash_burn': 8,
 'water': 15}

w_label_map={ 
 'clear': 10,
 'cloudy': 16,
 'haze': 6,
 'partly_cloudy': 13}

for f, tags in tqdm(df_train_f.values):
    img = cv2.imread(data_dir_haze.format(f))
    targets_f = np.zeros(13)
    for t in tags.split(' '):
        if t in f_label_map.keys:
            targets_f[f_label_map[t]] = 1 
    x_train_f.append(cv2.resize(img, (64, 64)))
    y_train_f.append(targets_f)
    
for f, tags in tqdm(df_train_w.values):
    img = cv2.imread(data_dir_nhaze.format(f))
    targets_w = np.zeros(4)
    for t in tags.split(' '):
        if t in w_label_map.keys:
            targets_w[w_label_map[t]] = 1 
    x_train_w.append(cv2.resize(img, (64, 64)))
    y_train_w.append(targets_w)

#for f, tags in tqdm(df_test.values):
#    img = cv2.imread('/media/gui/LENOVO/学习/AMAZON/test-jpg/{}.jpg'.format(f))
#    x_test.append(cv2.resize(img, (64, 64)))
    
y_train_f = np.array(y_train_f, np.uint8)
x_train_f = np.array(x_train_f, np.float32)/255.
y_train_w = np.array(y_train_f, np.uint8)
x_train_w = np.array(x_train_f, np.float32)/255.
#x_test  = np.array(x_test, np.float32)/255.

print(x_train_f.shape)
print(y_train_f.shape)

nfolds = 5

num_fold = 0
sum_score = 0

yfull_test = []
yfull_train =[]

kf = KFold(len(y_train_f), n_folds=nfolds, shuffle=True, random_state=1)

for train_index, test_index in kf:
        start_time_model_fitting = time.time() 
        X_train_f = x_train_f[train_index]
        Y_train_f = y_train_f[train_index]
        X_valid_f = x_train_f[test_index]
        Y_valid_f = y_train_f[test_index]
        num_fold += 1
        
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train_f), len(Y_train_f))
        print('Split valid: ', len(X_valid_f), len(Y_valid_f))
        
        kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')
        
        model = Sequential()
        model.add(BatchNormalization(input_shape=(64, 64,3)))
        model.add(Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(256, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(17, activation='sigmoid'))

        epochs_arr = [20, 5, 5]
        learn_rates = [0.001, 0.0001, 0.00001]

        for learn_rate, epochs in zip(learn_rates, epochs_arr):
            opt  = optimizers.Adam(lr=learn_rate)
            model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                          optimizer=opt,
                          metrics=['accuracy'])
            callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
            ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

            model.fit(x = X_train, y= Y_train, validation_data=(X_valid, Y_valid),
                  batch_size=128,verbose=2, epochs=epochs,callbacks=callbacks,shuffle=True)
        
        if os.path.isfile(kfold_weights_path):
            model.load_weights(kfold_weights_path)
        
        p_valid = model.predict(X_valid, batch_size = 128, verbose=2)
        print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

        p_train = model.predict(x_train, batch_size =128, verbose=2)
        yfull_train.append(p_train)
        
        p_test = model.predict(x_test, batch_size = 128, verbose=2)
        yfull_test.append(p_test)

result = np.array(yfull_test[0])
for i in range(1, nfolds):
    result += np.array(yfull_test[i])
result /= nfolds
result = pd.DataFrame(result, columns = labels)
result

from tqdm import tqdm
thres = [0.07, 0.17, 0.2, 0.04, 0.23, 0.33, 0.24, 0.22, 0.1, 0.19, 0.23, 0.24, 0.12, 0.14, 0.25, 0.26, 0.16]
preds = []
for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > 0.2, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))
    
df_test['tags'] = preds
df_test.to_csv('submission_keras_5_fold_CV_0.9136_LB_0.913.csv', index=False)

## 0.913
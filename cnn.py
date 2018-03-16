import os
import numpy as np
import tensorflow as tf
import datetime
import pandas as pd
from tqdm import tqdm
import cv2
import time
from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score

dir = "./train-jpg/"
kp = 0.5 # dropout = 1-kp
train_num = 16000

df_train_raw = pd.read_csv('./train_v2.csv')
df_train = df_train_raw[0:train_num]
flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

def read_labels(filename):
    result_x = []
    result_y = []


    label_map = {'agriculture': 14,
                 'artisinal_mine': 5,
                 'bare_ground': 1,
                 'blooming': 3,
                 'blow_down': 0,
                 'clear': 10,
                 'cloudy': 16,
                 'conventional_mine': 2,
                 'cultivation': 4,
                 'habitation': 9,
                 'haze': 6,
                 'partly_cloudy': 13,
                 'primary': 7,
                 'road': 11,
                 'selective_logging': 12,
                 'slash_burn': 8,
                 'water': 15}

    for f, tags in tqdm(df_train.values):
        img = cv2.imread('./train-small/{}.jpg'.format(f))
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        result_x.append(cv2.resize(img, (64, 64)))
        result_y.append(targets)

    return result_x,result_y


x_train,y_train = read_labels("train_v2.csv")
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32)/255.
num_fold = 0
sum_score = 0
nfolds = 5
yfull_test = []
yfull_train =[]
kf = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=1)
miniterations = 100

for train_index, test_index in kf:
    start_time_model_fitting = time.time()

    X_train = x_train[train_index]
    Y_train = y_train[train_index]
    X_valid = x_train[test_index]
    Y_valid = y_train[test_index]

    num_fold += 1
    print('Start KFold number {} from {}'.format(num_fold, nfolds))
    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_valid), len(Y_valid))

    kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')

    # 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
    x = tf.placeholder(tf.float32, shape=[None, 64 , 64 , 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 17])

    # 定义第一个卷积层的variables和ops
    W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 3, 16], stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]))
    keep_prob = tf.placeholder(tf.float32)
    L1_conv = tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
    L1_relu = tf.nn.relu(L1_conv + b_conv1)
    L1_relu_dr = tf.nn.dropout(L1_relu, keep_prob)
    L1_pool = tf.nn.max_pool(L1_relu_dr, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 定义第二个卷积层的variables和ops
    W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]))
    L2_conv = tf.nn.conv2d(L1_pool, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    L2_relu = tf.nn.relu(L2_conv + b_conv2)
    L2_relu_dr = tf.nn.dropout(L2_relu, keep_prob)
    L2_pool = tf.nn.max_pool(L2_relu_dr, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 全连接层
    W_fc1 = tf.Variable(tf.truncated_normal([16 * 16*32, 512], stddev=0.01))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]))

    h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 16*32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout层
    W_fc2 = tf.Variable(tf.truncated_normal([512, 17], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[17]))

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # 定义优化器和训练op
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv))
    # 加入正则化
    train_step = tf.train.AdamOptimizer((1e-3)).minimize(cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Start the tensorflow session" + str(num_fold))
        # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）
        batch_size = 100
        iterations = miniterations

        # 执行训练迭代
        for it in range(iterations):
            t_pre = datetime.datetime.now()
            input_count = len(X_train)
            batches_count = int(input_count / batch_size)
            remainder = input_count % batch_size
            input_labels = Y_train
            for n in range(batches_count):
                train_step.run(feed_dict={x: X_train[batch_size*n:batch_size*(n+1)],
                                          y_:Y_train[batch_size*n:batch_size*(n+1)], keep_prob: kp})
                if n % 10 == 0:
                    t_past = datetime.datetime.now()
                    t_used = int((t_past - t_pre).seconds)
                    t_pre = datetime.datetime.now()
                    iterate_accuracy = cross_entropy.eval(feed_dict={x: X_train[batch_size * n:batch_size * (n + 1)],
                                                                y_: Y_train[batch_size * n:batch_size * (n + 1)],
                                                                keep_prob: 1.0})
                    print('iteration %d: ceLoss %s, using %d s' % (it + 1, iterate_accuracy, t_used))

            if remainder > 0:
                start_index = batches_count * batch_size
                train_step.run(feed_dict={x: X_train[batch_size*n:batch_size*n+remainder],
                                          y_:Y_train[batch_size*n:batch_size*n+remainder], keep_prob: kp})


            print("valid ceLoss %g" % cross_entropy.eval(feed_dict={x: X_valid, y_: Y_valid, keep_prob: 1.0}))


            Y_predict = sess.run(y_conv,feed_dict={x:X_valid,keep_prob: 1.0})
            print(fbeta_score(Y_valid, np.array(Y_predict) > 0.2, beta=2, average='samples'))


print("end")
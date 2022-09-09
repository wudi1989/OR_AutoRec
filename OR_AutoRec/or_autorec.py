import argparse
import sys
import os.path
import random
import tensorflow as tf
import time
import numpy as np
import scipy
import pandas as pd
from scipy.sparse import csr_matrix


class or_autorec():
    def __init__(self, sess, num_user, num_item, learning_rate=0.001, reg_rate=1, epoch=500, batch_size=512,
                 verbose=False, T=3):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.verbose = verbose
        self.T = T
        self.gamma = 4


    def build_network(self, hidden_neuron=500):

        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.keep_rate_net = tf.placeholder(tf.float32)
        self.keep_rate_input = tf.placeholder(tf.float32)

        V = tf.Variable(tf.random_normal([hidden_neuron, self.num_user], stddev=0.01))
        W = tf.Variable(tf.random_normal([self.num_user, hidden_neuron], stddev=0.01))

        mu = tf.Variable(tf.random_normal([hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.num_user], stddev=0.01))
        layer_1 = tf.nn.dropout(tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix)),self.keep_rate_net)

        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        self.loss = tf.reduce_sum(tf.multiply(tf.log(1 + tf.square((self.rating_matrix - self.layer_2) / self.gamma)), self.rating_matrix_mask)) + self.reg_rate * (tf.square(tf.norm(W)) + tf.square(tf.norm(V)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, train_data):

        self.num_training = self.num_item
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering

        for i in range(total_batch):
            if i == total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={self.rating_matrix: self.train_data[:, batch_set_idx],
                                               self.rating_matrix_mask: self.train_data_mask[:, batch_set_idx],
                                               self.keep_rate_net: 0.95
                                               })
    def test(self, test_data):

        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.keep_rate_net: 1})
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        count = 0

        for (u, i) in test_set:
            count += 1
            pred_rating_test = self.predict(u, i)
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        print("RMSE:" + str(self.RMSE(error, count)) + "; MAE:" + str(self.MAE(error_mae, count)))
        self.min_rmse = min(self.RMSE(error, count), self.min_rmse)
        self.min_mae = min(self.MAE(error_mae, count), self.min_mae)
        print('min_rmse：' + str(self.min_rmse) + '  min_mae :' + str(self.min_mae))
        return self.RMSE(error, count), self.MAE(error_mae, count)

    def execute(self, train_data, test_data):
        mr = 0
        ir = 0.25
        # add_malicious_user
        # self.train_data = self.add_malicious_user(train_data, user_reg, ir=0.25)
        self.train_data = self._data_process(train_data)
        self.build_network()
        self.train_data_mask = scipy.sign(self.train_data)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.min_rmse = 10.000
        self.min_mae = 10.000
        start_time = time.time()
        for epoch in range(self.epochs):
            if self.verbose:
                print("Epoch: %04d;" % (epoch))
            self.train(train_data)
            end_time = time.time()
            if (epoch) % self.T == 0:
                print("Epoch: %04d; " % (epoch) + ', cost：' + str(end_time - start_time))
                rmse, mae = self.test(test_data)
        print("or_autorec." + '，learning rate： ' + str(self.learning_rate) + ', r：' + str(self.reg_rate) + ', num_user: ' + str(
            self.num_user) + ', num_item: ' + str(self.num_item) + ', mr: ' + str(mr) + ', ir：' + str(ir)  + ', min_rmse：' + str(self.min_rmse) + ',  min_mae :' + str(
            self.min_mae) + ', gamma:' + str(self.gamma))

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return self.reconstruction[user_id, item_id]

    def _data_process(self, data):
        output = np.zeros((self.num_user, self.num_item))
        for u in range(self.num_user):
            for i in range(self.num_item):
                output[u, i] = data.get((u, i))
        return output

    def add_malicious_user(self, data, reg, tmp):

        num = self.num_user
        self.num_user += int(self.num_user * reg)
        output = np.zeros((self.num_user, self.num_item))
        count = 0
        for u in range(self.num_user):

            for i in range(self.num_item):
                if u >= num:
                    ran = random.random()
                    if ran < tmp:
                        count += 1
                        tmp2 = random.random()
                        if tmp2 < 0.5:
                            output[u, i] = 1
                        else:
                            output[u, i] = 5
                else:
                    output[u, i] = data.get((u, i))
        return output

    def RMSE(self,error, num):
        return np.sqrt(error / num)

    def MAE(self,error_mae, num):
        return (error_mae / num)

def data_load(path1='../../Data/ml100k/temp/ml100k_train.csv',
                           path2='../../Data/ml100k/temp/ml100k_test.csv',
                           header=['user_id', 'item_id', 'rating','category']):
    train_data = pd.read_csv(path1, usecols=header)
    test_data = pd.read_csv(path2, usecols=header)

    n_users = max(train_data.user_id) if max(train_data.user_id) > max(test_data.user_id) else max(test_data.user_id)
    n_items = max(train_data.item_id) if max(train_data.item_id) > max(test_data.item_id) else max(test_data.item_id)


    train_row = []
    train_col = []
    train_rating = []
    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1.0
        train_row.append(u)
        train_col.append(i)
        train_rating.append(line[3])
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(line[3])
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    print("Load Data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_matrix.todok(), n_users, n_items


def parse_args():
    parser = argparse.ArgumentParser(description='OR-AutoRec')
    parser.add_argument('--model',  default='or_autorec')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--reg_rate', type=float, default=1)
    parser.add_argument('--T', type=int, default=3)


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    reg_rate = args.reg_rate


    train_data, test_data, n_user, n_item = data_load('./Data/ml100k/ml100k_train.csv',  './Data/ml100k/ml100k_test.csv')

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = or_autorec(sess, n_user, n_item)
        model.execute(train_data, test_data)
        sess.close()

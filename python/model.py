import tensorflow as tf
import h5py
import numpy as np
import scipy

import os
from os.path import join
from tqdm import tqdm
from utils import np_REG_batch, search_wav, wav2spec, spec2wav, copy_file


class REG:

    def __init__(self, log_path, saver_path, train_task, date, gpu_num, note):

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
        self.train_task = train_task
        self.log_path = log_path
        self.saver_path = saver_path
        self.saver_dir = '{}_{}/{}'.format(self.saver_path, note, date)

        self.saver_name = join(
            self.saver_dir, 'best_saver_{}'.format(self.train_task))
        self.tb_dir = '{}_{}/{}'.format(self.log_path, note, date)
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True

        if not os.path.exists(self.saver_dir):
            os.makedirs(self.saver_dir)
        if not os.path.exists(self.tb_dir):
            os.makedirs(self.tb_dir)

    def build(self, init_learning_rate, reuse):
        self.init_learning_rate = init_learning_rate
        self.name = 'REG_Net'
        # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            with tf.variable_scope('Intputs'):
                self.x_noisy = tf.placeholder(
                    tf.float32, shape=[None, 1285], name='x')
            with tf.variable_scope('Outputs'):
                self.y_clean = tf.placeholder(
                    tf.float32, shape=[None, 257], name='y_clean')
            with tf.name_scope('weights'):
                w = {'w_o': tf.get_variable("WO", shape=[512, 257],
                                           # regularizer=regularizer,
                                           initializer=tf.contrib.layers.xavier_initializer()),
                    'w_1': tf.get_variable("W1", shape=[1285, 512],
                                           # regularizer=regularizer,
                                           initializer=tf.contrib.layers.xavier_initializer()),
                    'w_2': tf.get_variable("W2", shape=[512, 512],
                                           # regularizer=regularizer,
                                           initializer=tf.contrib.layers.xavier_initializer()),
                    'w_3': tf.get_variable("W3", shape=[512, 512],
                                           # regularizer=regularizer,
                                           initializer=tf.contrib.layers.xavier_initializer()),
                    'w_4': tf.get_variable("W4", shape=[512, 512],
                                           # regularizer=regularizer,
                                           initializer=tf.contrib.layers.xavier_initializer())}
            with tf.name_scope('bias'):
                b = {'b_o': tf.get_variable("bO", shape=[1, 257],
                                           initializer=tf.constant_initializer(value=0, dtype=tf.float32)),
                    'b_1': tf.get_variable("b1", shape=[1, 512],
                                           initializer=tf.constant_initializer(value=0, dtype=tf.float32)),
                    'b_2': tf.get_variable("b2", shape=[1, 512],
                                           initializer=tf.constant_initializer(value=0, dtype=tf.float32)),
                    'b_3': tf.get_variable("b3", shape=[1, 512],
                                           initializer=tf.constant_initializer(value=0, dtype=tf.float32)),
                    'b_4': tf.get_variable("b4", shape=[1, 512],
                                           initializer=tf.constant_initializer(value=0, dtype=tf.float32))}
            with tf.variable_scope('DNN'):
                layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(self.x_noisy, w['w_1']), b['b_1']))
                layer_2 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_1, w['w_2']), b['b_2']))
                layer_3 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_2, w['w_3']), b['b_3']))
                layer_4 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_3, w['w_4']), b['b_4']))
                self.reg_layer = tf.add(tf.matmul(layer_4, w['w_o']), b['b_o'])


            with tf.name_scope('reg_loss'):

                self.loss_reg = tf.losses.mean_squared_error(
                    self.y_clean, self.reg_layer)

                tf.summary.scalar('Loss reg', self.loss_reg)
            
            with tf.name_scope("exp_learning_rate"):
                self.global_step = tf.Variable(0, trainable=False)
                self.exp_learning_rate = tf.train.exponential_decay(self.init_learning_rate,
                                                                    global_step=self.global_step,
                                                                    decay_steps=500000, decay_rate=0.95, staircase=False)
                tf.summary.scalar('Learning rate', self.exp_learning_rate)

            optimizer = tf.train.AdamOptimizer(self.init_learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(self.loss_reg))
            gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
            self.optimizer = optimizer.apply_gradients(zip(gradients, v),
                                                      global_step=self.global_step)
            self.saver = tf.train.Saver()


    def train(self, training_data_dir, split_num, epochs, batch_size):
        if tf.gfile.Exists(self.tb_dir):
            tf.gfile.DeleteRecursively(self.tb_dir)
            tf.gfile.MkDir(self.tb_dir)
        best_reg_loss = 10.

        with tf.Session(config=self.config) as sess:

            print('Start Training')
            # set early stopping
            patience = 10
            FLAG = False
            min_delta = 0.01
            step = 0
            epochs = range(epochs)
            
            tf.global_variables_initializer().run()
            writer = tf.summary.FileWriter(
                self.tb_dir, sess.graph,  max_queue=10)
            merge_op = tf.summary.merge_all()
            for epoch in tqdm(epochs):
                shuffle_list = np.arange(split_num)
                np.random.shuffle(shuffle_list)
                loss_reg_tmp = 0.
                count = 0
                for i in tqdm(shuffle_list):
                    data_name = join(training_data_dir,
                                     '{}_{}.h5'.format(self.train_task, i))
                    data_file = h5py.File(data_name, 'r')
                    clean_data = data_file['clean_data']
                    noisy_data = data_file['noisy_data']
                    data_len = len(clean_data)
                    data_batch = np_REG_batch(
                        noisy_data, clean_data, batch_size, data_len)
                    for batch in range(int(data_len / batch_size)):
                        noisy_batch, clean_batch = next(
                            data_batch), next(data_batch)
                        feed_dict = {self.x_noisy: noisy_batch,
                                     self.y_clean: clean_batch}
                        _, loss_var1, summary_str = sess.run(
                            [self.optimizer, self.loss_reg, merge_op], feed_dict=feed_dict)
                        loss_reg_tmp += loss_var1
                        count += 1
                        writer.add_summary(summary_str, step)
                        step += 1
                if epoch % 1 == 0:
                    loss_reg_tmp /= count
                    loss_var = loss_reg_tmp

                    print('[epoch {}] Loss Reg:{}'.format(
                        int(epoch), loss_reg_tmp))
                    if loss_var <= (best_reg_loss - min_delta):
                        best_reg_loss = loss_var
                        self.saver.save(sess=sess, save_path=self.saver_name)
                        patience = 10
                        print('Best Reg Loss: ', best_reg_loss)
                    else:
                        print('Not improve Loss:', best_reg_loss)
                        if FLAG == True:
                            patience -= 1
                if patience == 0 and FLAG == True:
                    print('Early Stopping ! ! !')
                    break

    def test(self, testing_data_dir, result_dir, test_saver, n_cores, num_test=False):
        print('Start Testing')
        tmp_list = search_wav(testing_data_dir)

        if num_test:
            test_list = np.random.choice(tmp_list, num_test)
        else:
            test_list = tmp_list

        print('All testing data number:', len(test_list))
        REG_dir = join(result_dir, 'REG')
        Noisy_write_dir = join(result_dir, 'Source')
        Clean_write_dir = join(result_dir, 'Target')


        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            os.makedirs(REG_dir)
            os.makedirs(Noisy_write_dir)
            os.makedirs(Clean_write_dir)
        with tf.Session(config=self.config) as sess:
            self.saver.restore(sess=sess, save_path=test_saver)
            for file in tqdm(test_list):
                hop_length = 256
                file_name = file.split('/')[-1]
                try:
                    snr, noise_name, clean_name1, clean_neme2 = file.split('/')[-1].split('_')
                    clean_file = join(testing_data_dir, '_'.join(['0dB', 'n0', clean_name1, clean_neme2]))
                    noisy_file = file
                except:
                    snr, noise_name, clean_name = file.split('/')[-1].split('_')
                noisy_file = join(testing_data_dir, file_name)
                REG_file = join(REG_dir, file_name)
                Noisy_file = join(Noisy_write_dir, file_name)
                Clean_file = join(Clean_write_dir, file_name)

                X_in_seq = wav2spec(noisy_file, sr=16000,
                                     forward_backward=True, SEQUENCE=False, norm=True, hop_length=hop_length)
                re_reg = sess.run([self.reg_layer],
                                  feed_dict={self.x_noisy: X_in_seq})[:][0]
                spec2wav(noisy_file, 16000, REG_file,
                         re_reg, hop_length=hop_length)
                copy_file(noisy_file, Noisy_file)
                copy_file(clean_file, Clean_file)

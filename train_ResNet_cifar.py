import tensorflow as tf
import numpy as np
import os
import pickle
import datetime
from config import config_ResNet as config
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


class ResNet(object):
    def __init__(self, is_train=True):
        # Placeholders for input, output and dropout
        self.image_size = 32
        self.summary_step = 1000
        self.num_classes = 10
        self.dropout_keep_prob = 0.5
        self.initial_learning_rate = 0.1
        self.decay_steps = 16000
        self.decay_rate = 0.1
        self.staircase = True
        self.epsilon = 1e-3
        self.decay = 0.99
        self.is_train = is_train
        self.weight_decay = 0.00004

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_x = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name="input_x")
            self.label = tf.placeholder(tf.int32, [None, self.num_classes], name="label")
            if is_train:
                # self.input_x = tf.image.random_flip_left_right(self.input_x)
                self.input_x = tf.image.resize_image_with_crop_or_pad(
                    tf.pad(self.input_x, np.array([[0, 0], [2, 2], [2, 2], [0, 0]]), name='random_crop_pad'),
                    self.image_size, self.image_size)

            self.logits = self.build_network('ResNet')

            ConfigProto = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))
            ConfigProto.gpu_options.allow_growth = True
            self.sess = tf.Session(config=ConfigProto, graph=self.graph)
            self.saver = tf.train.Saver(max_to_keep=40)

            if self.is_train:
                # self.label = tf.placeholder(tf.int32, [None, self.num_classes], name="label")
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                # self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
                self.learning_rate = tf.train.exponential_decay(
                    self.initial_learning_rate, self.global_step, self.decay_steps,
                    self.decay_rate, self.staircase, name='learning_rate')
                self.loss, self.accuracy = self.compute_loss()

                self.summary_writer = tf.summary.FileWriter(config.summary_dir, graph=self.sess.graph)
                # for BN
                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(extra_update_ops):
                    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                    # self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,
                                                                                          global_step=self.global_step,
                                                                                          name='optimizer')

                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar("acc", self.accuracy)
                tf.summary.scalar("lr", self.learning_rate)
                self.summary_op = tf.summary.merge_all()

            filename = tf.train.latest_checkpoint(config.checkpoint_dir)
            self.sess.run(tf.global_variables_initializer())
            if filename is not None:
                print('restore from : ', filename)
                self.saver.restore(self.sess, filename)

    def get_variable_wd(self, name, shape, trainable=True):
        vars= tf.get_variable(name, shape=shape,
                            initializer=tf.truncated_normal_initializer(),
                            # regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay),
                            trainable=trainable)
        weight_decay = tf.nn.l2_loss(vars) * self.weight_decay
        tf.add_to_collection('losses', weight_decay)
        return vars

    def conv_layer(self, name, inputs, filters, size, stride=1, padding='SAME', activate=None):
        with tf.variable_scope(name):
            # weight = tf.get_variable('conv_weight', shape=[size, size, int(inputs.shape[3]), filters],
            #                          initializer=tf.truncated_normal_initializer(),
            #                          trainable=True)
            # bias = tf.get_variable('conv_bias', shape=filters,
            #                        initializer=tf.truncated_normal_initializer(),
            #                        trainable=True)
            weight = self.get_variable_wd('conv_weight', shape=[size, size, int(inputs.shape[3]), filters])
            # bias = self.get_variable_wd('conv_bias', shape=[filters])
            print(inputs)
            print(weight)
            conv = tf.nn.conv2d(inputs, weight, strides=[1, stride, stride, 1], padding=padding,
                                name='conv')
            # conv_biased = tf.add(conv, bias, name='conv_biased')
            # return conv_biased if activate is None else tf.nn.relu(conv_biased)
            return conv if activate is None else tf.nn.relu(conv)


    def batch_norm_conv(self, name, inputs, activate=None):
        # https://git.alphagriffin.com/O.P.P/FiryZeplin-deep-learning/src/ee5b04a3ff360d8276d881e41265d7d45f47ccc9/batch-norm/Batch_Normalization_Solutions.ipynb
        with tf.variable_scope(name):
            batch_mean, batch_variance = tf.nn.moments(inputs, axes=[0, 1, 2], keep_dims=False)
            beta = tf.Variable(tf.constant(0.0, shape=[int(inputs.shape[3])]), trainable=True, dtype=tf.float32,
                               name="beta")
            gamma = tf.Variable(tf.constant(1.0, shape=[int(inputs.shape[3])]), trainable=True, dtype=tf.float32,
                                name="gamma")
            pop_mean = tf.Variable(tf.zeros([int(inputs.shape[3])]), trainable=False)
            pop_variance = tf.Variable(tf.ones([int(inputs.shape[3])]), trainable=False)

            train_mean = tf.assign(pop_mean, pop_mean * self.decay + batch_mean * (1 - self.decay))
            train_variance = tf.assign(pop_variance, pop_variance * self.decay + batch_variance * (1 - self.decay))

            tf.summary.histogram('train_mean', train_mean)
            tf.summary.histogram('train_variance', train_variance)
            tf.summary.histogram('beta', beta)
            tf.summary.histogram('gamma', gamma)

            if self.is_train:
                with tf.control_dependencies([train_mean, train_variance]):
                    bn = tf.nn.batch_normalization(inputs, batch_mean, batch_variance, beta, gamma, self.epsilon, name='BatchNorm')
            else:
                bn = tf.nn.batch_normalization(inputs, pop_mean, pop_variance, beta, gamma, self.epsilon, name='BatchNorm')

            return bn if activate is None else tf.nn.relu(bn)

    def batch_norm_fc(self, name, inputs, activate=None):
        with tf.variable_scope(name):
            batch_mean, batch_variance = tf.nn.moments(inputs, axes=[0])
            beta = tf.Variable(tf.constant(0.0, shape=[int(inputs.shape[1])]), trainable=True, dtype=tf.float32,
                               name="beta")
            gamma = tf.Variable(tf.constant(1.0, shape=[int(inputs.shape[1])]), trainable=True, dtype=tf.float32,
                                name="gamma")
            pop_mean = tf.Variable(tf.zeros([int(inputs.shape[1])]), trainable=False)
            pop_variance = tf.Variable(tf.ones([int(inputs.shape[1])]), trainable=False)

            train_mean = tf.assign(pop_mean, pop_mean * self.decay + batch_mean * (1 - self.decay))
            train_variance = tf.assign(pop_variance, pop_variance * self.decay + batch_variance * (1 - self.decay))

            tf.summary.histogram('batch_mean', batch_mean)
            tf.summary.histogram('batch_variance', batch_variance)
            tf.summary.histogram('beta', beta)
            tf.summary.histogram('gamma', gamma)

            if self.is_train:
                with tf.control_dependencies([train_mean, train_variance]):
                    bn = tf.nn.batch_normalization(inputs, batch_mean, batch_variance, beta, gamma, self.epsilon, name='BatchNorm')

            else:
                bn = tf.nn.batch_normalization(inputs, pop_mean, pop_variance, beta, gamma, self.epsilon, name='BatchNorm')

            return bn if activate is None else tf.nn.relu(bn)

    def max_pooling(self, name, inputs, size, stride=1, padding='SAME'):
        with tf.variable_scope(name):
            return tf.nn.max_pool(
                inputs,
                ksize=[1, size, size, 1],
                strides=[1, stride, stride, 1],
                padding=padding,
                name="max_pool")

    def gobal_avg_pool(self, name, inputs, stride=1):
        with tf.variable_scope(name):
            return tf.nn.avg_pool(
                inputs,
                ksize=[1, int(inputs.shape[1]), int(inputs.shape[2]), 1],
                strides=[1, stride, stride, 1],
                padding='VALID',
                name="avg_pool")

    def fc_layer(self, name, inputs, outputs, activate=None, trainable=True):
        with tf.variable_scope(name):
            # weight = tf.get_variable('fc_weight', shape=[int(inputs.shape[1]), outputs],
            #                          initializer=tf.truncated_normal_initializer(),
            #                          trainable=trainable)
            # bias = tf.get_variable('fc_bias', shape=outputs,
            #                        initializer=tf.truncated_normal_initializer(),
            #                        trainable=trainable)

            weight = self.get_variable_wd('fc_weight', shape=[int(inputs.shape[1]), outputs])
            bias = self.get_variable_wd('fc_bias', shape=outputs)

            return tf.add(tf.matmul(inputs, weight), bias) if activate is None else tf.nn.relu(
                tf.add(tf.matmul(inputs, weight), bias))

    def residual_block(self, name, inputs, filters, size, stride=1):
        with tf.variable_scope(name):
            # sub sampling은 첫번째 conv 에서 stride 2로 진행
            # The subsampling is performed by convolutions with a stride of 2
            block = tf.pad(inputs, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_1')
            block = self.conv_layer('residual_block_conv_1', block, filters=filters, size=size, stride=stride, padding='VALID')
            block = self.batch_norm_conv('residual_block_bn_1', block, activate='relu')
            block = self.conv_layer('residual_block_conv_2' , block, filters=filters, size=size, stride=1, padding='SAME')
            block = self.batch_norm_conv('residual_block_bn_2', block, activate=None)

            if int(block.shape[1]) != int(inputs.shape[1]):
                inputs = tf.pad(inputs, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_1')
                inputs = self.conv_layer('residual_block_conv_projection', inputs, filters=filters, size=size, stride=stride, padding='VALID')

            block += inputs

            return tf.nn.relu(block)


    def build_network(self, name):
        with tf.variable_scope(name):
            # https://towardsdatascience.com/resnets-for-cifar-10-e63e900524e0
            model = self.conv_layer('conv_1', self.input_x, filters=16, size=1, stride=1, padding='SAME')
            model = self.batch_norm_conv('bn_1', model)
            print('conv_1 :  ', model)

            # 첫번째 residual block 만 stride 1
            model = self.residual_block('layers_2n', model, filters=16, size=3, stride=1)
            print('layers_1 : ', model)
            model = self.residual_block('layers_4n', model, filters=32, size=3, stride=2)
            print('layers_2 : ', model)
            model = self.residual_block('layers_6n', model, filters=64, size=3, stride=2)
            print('layers_3 : ', model)
            model =self.gobal_avg_pool('gobal_avg_pool', model)
            print('gobal_avg_pool : ', model)
            model = tf.reshape(model, [-1, int(model.shape[1]) * int(model.shape[2]) * int(model.shape[3])])
            model = self.fc_layer('fc', model, self.num_classes, activate=None)
            print('fc_layer : ', model)

            return model

    def compute_loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label))
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.label))
        tf.add_to_collection('losses', loss)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.label, axis=1)), tf.float32),
            name='accuracy')

        return tf.add_n(tf.get_collection('losses'), name='total_loss'), accuracy


    def train(self, batch_x, batch_y, save=False):
        # print(self.graph)
        with self.graph.as_default():
            feed_dict = {self.input_x: batch_x,
                           self.label: batch_y}

            _, global_step, summary_str, loss, acc = self.sess.run(
                [self.train_op, self.global_step, self.summary_op, self.loss, self.accuracy],
                feed_dict=feed_dict)
            if save:
                self.summary_writer.add_summary(summary_str, global_step=global_step)
                self.saver.save(self.sess, os.path.join(config.checkpoint_dir, 'model.ckpt'), global_step=global_step)

            return global_step, loss, acc

    def validate(self, batch_x, batch_y):
        with self.graph.as_default():
            feed_dict = {self.input_x: batch_x,
                         self.label: batch_y}
            # ox, accuracy, loss = self.sess.run([self.ox, self.accuracy, self.total_loss], feed_dict=feed_dict)
            loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            return loss, acc

    def predcit(self, batch_x, batch_y):
        with self.graph.as_default():
            equal = tf.reduce_sum(
                tf.cast(tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.label, axis=1)), tf.float32))

            feed_dict = {self.input_x: batch_x,
                         self.label: batch_y}
            # ox, accuracy, loss = self.sess.run([self.ox, self.accuracy, self.total_loss], feed_dict=feed_dict)
            equal = self.sess.run([equal], feed_dict=feed_dict)
            return equal



if __name__ == '__main__':
    train_x = None
    train_y = None
    n_class = 10
    batch_size = 128
    epochs = 100

    folder_name = "./data_set/cifar_10"
    for i in range(1,6):
        f = open(os.path.join(folder_name, 'data_batch_' + str(i)), 'rb')
        datadict = pickle.load(f, encoding='latin1')

        datas = datadict["data"]
        labels = np.array(datadict['labels'])
        labels = np.eye(n_class)[labels]

        datas = datas / 255.0
        datas = datas.reshape([-1, 3, 32, 32])
        datas = datas.transpose([0, 2, 3, 1])

        if train_x is None:
            train_x = datas
            train_y = labels
        else:
            train_x = np.concatenate((train_x, datas), axis=0)
            train_y = np.concatenate((train_y, labels), axis=0)

        f.close()


    f = open(os.path.join(folder_name, 'test_batch'), 'rb')
    datadict = pickle.load(f, encoding='latin1')
    f.close()

    test_x = datadict["data"]
    test_y = np.array(datadict['labels'])
    test_y = np.eye(n_class)[test_y]

    test_x = np.array(test_x) / 255.0
    test_x = test_x.reshape([-1, 3, 32, 32])
    test_x = test_x.transpose([0, 2, 3, 1])




    net = ResNet(True)
    train_set_len = train_x.shape[0]
    r_idx = np.arange(train_x.shape[0])
    total_batch = int(train_set_len / batch_size)
    for epoch in range(epochs):
        # print(train_x[0])

        r_idx = np.arange(train_x.shape[0])
        np.random.shuffle(r_idx)
        train_x = train_x[r_idx]
        train_y = train_y[r_idx]

        for i in range(total_batch+1):
            if ((i + 1) * batch_size) > train_set_len:
                break

            batch_x = train_x[i * batch_size: (i+1) * batch_size]
            batch_y = train_y[i * batch_size: (i+1) * batch_size]
            if i % 100 == 0:
                global_step, train_loss, train_acc = net.train(batch_x, batch_y, True)
                val_loss, val_acc = net.validate(test_x[:200], test_y[:200])
                print('%d step\ttrain loss : %.3f\ttrain accuracy : %.3f\tval loss : %.3f\tval accuracy : %.3f'%(global_step, train_loss, train_acc, val_loss, val_acc))
            else:
                _, loss, ac = net.train(batch_x, batch_y, False)


    tf.reset_default_graph()
    net = ResNet(False)
    test_set_len = test_x.shape[0]
    total_batch = int(test_set_len / batch_size)
    total_equal = 0
    for i in range(total_batch + 1):
        if ((i + 1) * batch_size) > test_set_len:
            break

        batch_x = test_x[i * batch_size: (i + 1) * batch_size]
        batch_y = test_y[i * batch_size: (i + 1) * batch_size]

        equal = net.predcit(batch_x, batch_y)
        total_equal += equal[0]

    print('test accuracy : %.3f'%(total_equal/test_set_len))


        # minist
        # for i in range(total_batch):
        #     batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        #     batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        #     if i % 100 == 0:
        #         print(i, 'step')
        #         loss, acc = net.train(batch_xs, batch_ys, True)
        #         print('train : ', loss, acc)
        #         batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        #         batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        #         loss, acc = net.validate(batch_xs, batch_ys)
        #         print('val : ', loss, acc)
        #     else:
        #         loss, acc = net.train(batch_xs, batch_ys, False)

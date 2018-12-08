import tensorflow as tf
import tensorflow.contrib as tfc
import tensorflow.contrib.keras as keras
from .model_utils import *


class Model:
    def __init__(self, sess, height=256, weight=512, batch_size=4, max_disp=129, lr=0.001, cnn_3d_type='normal'):
        self.reg = 1e-4  # regularization factor
        self.max_disp = max_disp
        self.image_size_tf = None
        self.height = height
        self.weight = weight
        self.batch_size = batch_size
        self.lr = lr
        self.cnn_3d = cnn_3d_type
        self.sess = sess
        self.build_model()

    def build_model(self):
        self.left = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3])
        self.right = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3])
        self.label = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight])
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.image_size_tf = tf.shape(self.left)[1:3]

        conv4_left = self.CNN(self.left)
        fusion_left = self.SPP(conv4_left)

        conv4_right = self.CNN(self.right, True)
        fusion_right = self.SPP(conv4_right, True)

        cost_vol = self.cost_vol(fusion_left, fusion_right, self.max_disp)
        if self.cnn_3d == 'normal':
            outputs = self.CNN3D(cost_vol)
        elif self.cnn_3d == 'resnet_3d':
            outputs = self.resnet_3d(cost_vol)
        elif self.cnn_3d == 'densenet_3d':
            outputs = self.densenet_3d(cost_vol)
        else:
            raise ValueError('Does not support {}'.format(self.cnn_3d))
        self.disps = self.output(outputs)

        # only compute valid labeled points
        disps_mask = tf.where(self.label > 0., self.disps, self.label)

        self.loss = self._smooth_l1_loss(disps_mask, self.label)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        self.train_op = optimizer.minimize(self.loss)
        try:
          self.sess.run(tf.global_variables_initializer())
        except:
          self.sess.run(tf.initialize_all_variables())

    def _smooth_l1_loss(self, disps_pred, disps_targets, sigma=1.0):
        sigma_2 = sigma ** 2
        box_diff = disps_pred - disps_targets
        in_box_diff = box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box))
        return loss_box

    def CNN(self, bottom, reuse=False):
        with tf.variable_scope('CNN'):
            with tf.variable_scope('conv0'):
                bottom = conv_block(tf.layers.conv2d, bottom, 32, 3, strides=2, name='conv0_1', reuse=reuse, reg=self.reg)
                for i in range(1, 3):
                    bottom = conv_block(tf.layers.conv2d, bottom, 32, 3, name='conv0_%d' % (i+1), reuse=reuse, reg=self.reg)
            with tf.variable_scope('conv1'):
                for i in range(3):
                    bottom = res_block(tf.layers.conv2d, bottom, 32, 3, name='conv1_%d' % (i+1), reuse=reuse, reg=self.reg)
            with tf.variable_scope('conv2'):
                bottom = res_block(tf.layers.conv2d, bottom, 64, 3, strides=2, name='conv2_1', reuse=reuse, reg=self.reg,
                                   projection=True)
                for i in range(1, 8):
                    bottom = res_block(tf.layers.conv2d, bottom, 64, 3, name='conv2_%d' % (i+1), reuse=reuse, reg=self.reg)
            with tf.variable_scope('conv3'):
                bottom = res_block(tf.layers.conv2d, bottom, 128, 3, dilation_rate=2, name='conv3_1', reuse=reuse,
                                   reg=self.reg, projection=True)
                for i in range(1, 3):
                    bottom = res_block(tf.layers.conv2d, bottom, 128, 3, dilation_rate=2, name='conv3_%d' % (i+1), reuse=reuse,
                                       reg=self.reg)
            with tf.variable_scope('conv4'):
                for i in range(3):
                    bottom = res_block(tf.layers.conv2d, bottom, 128, 3, dilation_rate=4, name='conv4_%d' % (i+1), reuse=reuse,
                                       reg=self.reg)
        return bottom

    def SPP(self, bottom, reuse=False):
        with tf.variable_scope('SPP'):
            branches = []
            for i, p in enumerate([64, 32, 16, 8]):
                branches.append(SPP_branch(tf.layers.conv2d, bottom, p, 32, 3, name='branch_%d' % (i+1), reuse=reuse,
                                           reg=self.reg))
            conv2_16 = tf.get_default_graph().get_tensor_by_name('CNN/conv2/conv2_8/add:0')
            conv4_3 = tf.get_default_graph().get_tensor_by_name('CNN/conv4/conv4_3/add:0')
            concat = tf.concat([conv2_16, conv4_3] + branches, axis=-1, name='concat')
            with tf.variable_scope('fusion'):
                bottom = conv_block(tf.layers.conv2d, concat, 128, 3, name='conv1', reuse=reuse, reg=self.reg)
                fusion = conv_block(tf.layers.conv2d, bottom, 32, 1, name='conv2', reuse=reuse, reg=self.reg)
        return fusion

    def cost_vol(self, left, right, max_disp=192):
        with tf.variable_scope('cost_vol'):
            shape = tf.shape(right)
            right_tensor = keras.backend.spatial_2d_padding(right, padding=((0, 0), (max_disp // 4, 0)))
            disparity_costs = []
            for d in reversed(range(max_disp // 4)):
                left_tensor_slice = left
                right_tensor_slice = tf.slice(right_tensor, begin=[0, 0, d, 0], size=shape)
                right_tensor_slice.set_shape(tf.TensorShape([None, None, None, 32]))
                cost = tf.concat([left_tensor_slice, right_tensor_slice], axis=3)
                disparity_costs.append(cost)
            cost_vol = tf.stack(disparity_costs, axis=1)
        return cost_vol

    def CNN3D(self, bottom):
        with tf.variable_scope('CNN3D'):
            for i in range(6):
                bottom = conv_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv0_%d' % (i+1), reg=self.reg)

            out = conv_block(tf.layers.conv3d, bottom, 1, 3, name='3Dconv5', reg=self.reg)
        return out

    def resnet_3d(self, bottom):
        with tf.variable_scope('RES3D'):
            for i in range(2):
                bottom = conv_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv0_%d' % (i+1), reg=self.reg)

            _3Dconv1 = res_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv1', reg=self.reg)
            _3Dconv1 = res_block(tf.layers.conv3d, _3Dconv1, 32, 3, name='3Dconv2', reg=self.reg)
            _3Dconv1 = res_block(tf.layers.conv3d, _3Dconv1, 32, 3, name='3Dconv3', reg=self.reg)
            _3Dconv1 = res_block(tf.layers.conv3d, _3Dconv1, 32, 3, name='3Dconv4', reg=self.reg)
            out = conv_block(tf.layers.conv3d, _3Dconv1, 1, 3, name='3Dconv5', reg=self.reg)
        return out

    def densenet_3d(self, bottom):
        with tf.variable_scope('DENSE3D'):
            for i in range(2):
                bottom = conv_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv0_%d' % (i+1), reg=self.reg)

            out = [bottom]
            for j in range(5):
                bottom = conv_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv1_%d' % (j+1), reg=self.reg)
                out.append(bottom)
                bottom = tf.concat(out, axis=-1)

            out = conv_block(tf.layers.conv3d, bottom, 1, 3, name='3Dconv5', reg=self.reg)
        return out

    def output(self, output):
        squeeze = tf.squeeze(output, [4])
        transpose = tf.transpose(squeeze, [0, 2, 3, 1])

        upsample = tf.transpose(tf.image.resize_images(transpose, self.image_size_tf), [0, 3, 1, 2])
        upsample = tf.image.resize_images(upsample, tf.constant([self.max_disp, self.height], dtype=tf.int32))
        disps = soft_arg_min(upsample, 'soft_arg_min')
        return disps

    def train(self, left, right, label):
        _, loss = self.sess.run([self.train_op, self.loss],
                                feed_dict={self.left: left, self.right: right, self.label: label, self.is_training:True})
        return loss

    def test(self, left, right, label):
        pred, loss = self.sess.run([self.disps, self.loss],
                                   feed_dict={self.left: left, self.right: right, self.label: label, self.is_training:False})
        return pred, loss

    def predict(self, left, right):
        pred = self.sess.run([self.disps],
                             feed_dict={self.left: left, self.right: right, self.is_training:False})
        return pred

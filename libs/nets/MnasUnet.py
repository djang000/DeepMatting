import tensorflow as tf
import tensorflow.contrib.slim as slim
slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

class MnasUnet(object):
    def __init__(self, input, num_classes=1000, weight_decay=0.0004, is_training=True):
        self.imgs      = input
        self.depth_multiplier = 1.0
        self.min_depth = 8
        self.num_classes = num_classes
        self.up_sample = 3
        self.weight_decay = weight_decay
        self.is_training = is_training
        self.end_points = {}
        self.end_points['input'] = input
        self.build_model()

    def build_model(self):
        arg_scope = training_scope(is_training=self.is_training)
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('MnasNet') as sc:
                depth = lambda d: max(int(d * self.depth_multiplier), self.min_depth)
                x = slim.conv2d(self.imgs, depth(32), [3, 3], stride=2, activation_fn=tf.nn.relu6)
                x = self.InvertedBottleneck(x, kernel_size=[3, 3], up_sample=1, out_ch=16, stride=1,
                                                scope='expanded_conv')
                s1 = x

                x = self.InvertedBottleneck(x, kernel_size=[3, 3], up_sample=self.up_sample, out_ch=24, stride=2,
                                            scope='expanded_conv_1')
                x = self.InvertedBottleneck(x, kernel_size=[3, 3], up_sample=self.up_sample, out_ch=24, stride=1,
                                            scope='expanded_conv_2')
                x = self.InvertedBottleneck(x, kernel_size=[3, 3], up_sample=self.up_sample, out_ch=24, stride=1,
                                            scope='expanded_conv_3')
                s2 = x

                x = self.InvertedBottleneck(x, kernel_size=[5, 5], up_sample=self.up_sample, out_ch=40, stride=2,
                                            scope='expanded_conv_4')
                x = self.InvertedBottleneck(x, kernel_size=[5, 5], up_sample=self.up_sample, out_ch=40, stride=1,
                                            scope='expanded_conv_5')
                x = self.InvertedBottleneck(x, kernel_size=[5, 5], up_sample=self.up_sample, out_ch=40, stride=1,
                                            scope='expanded_conv_6')
                s3 = x

                self.up_sample=6
                x = self.InvertedBottleneck(x, kernel_size=[5, 5], up_sample=self.up_sample, out_ch=80, stride=2,
                                            scope='expanded_conv_7')
                x = self.InvertedBottleneck(x, kernel_size=[5, 5], up_sample=self.up_sample, out_ch=80, stride=1,
                                            scope='expanded_conv_8')
                x = self.InvertedBottleneck(x, kernel_size=[5, 5], up_sample=self.up_sample, out_ch=80, stride=1,
                                            scope='expanded_conv_9')

                x = self.InvertedBottleneck(x, kernel_size=[3, 3], up_sample=self.up_sample, out_ch=96, stride=1,
                                            scope='expanded_conv_10')
                x = self.InvertedBottleneck(x, kernel_size=[3, 3], up_sample=self.up_sample, out_ch=96, stride=1,
                                            scope='expanded_conv_11')
                s4 = x

                x = self.InvertedBottleneck(x, kernel_size=[5, 5], up_sample=self.up_sample, out_ch=192, stride=2,
                                            scope='expanded_conv_12')
                x = self.InvertedBottleneck(x, kernel_size=[5, 5], up_sample=self.up_sample, out_ch=192, stride=1,
                                            scope='expanded_conv_13')
                x = self.InvertedBottleneck(x, kernel_size=[5, 5], up_sample=self.up_sample, out_ch=192, stride=1,
                                            scope='expanded_conv_14')
                x = self.InvertedBottleneck(x, kernel_size=[5, 5], up_sample=self.up_sample, out_ch=192, stride=1,
                                            scope='expanded_conv_15')

                x = self.InvertedBottleneck(x, kernel_size=[3, 3], up_sample=self.up_sample, out_ch=320, stride=1,
                                            scope='expanded_conv_16')
                s5 = x

            with tf.variable_scope('Unet'):
                up1 = self.UpConv(s5, s4, 192, 1)
                up2 = self.UpConv(up1, s3, 96, 2)
                up3 = self.UpConv(up2, s2, 48, 3)
                up = self.UpConv(up3, s1, 24, 4)
                shape = self.imgs.get_shape().as_list()[1:3]
                resize_feature = tf.image.resize_bilinear(up, shape)
                binary_mask = slim.conv2d(resize_feature, 2, [1, 1], stride=1, activation_fn=tf.nn.sigmoid)

            with tf.variable_scope('feathering'):
                rbg, rfg = tf.split(binary_mask, num_or_size_splits=2, axis=3)
                square_input = self.imgs * self.imgs
                mask_img = self.imgs * rfg
                feather_input = tf.concat([self.imgs, square_input, mask_img, binary_mask], axis=3)

                f_conv1 = slim.conv2d(feather_input, 11, [3, 3], stride=1, activation_fn=tf.nn.relu6, scope='conv1')

                feathers = slim.conv2d(f_conv1, 3, [3, 3], stride=1, normalizer_fn=None, normalizer_params=None, activation_fn=None, scope='conv2')

                a, b, c = tf.split(feathers, num_or_size_splits=3, axis=3)
                sum_a = a * rbg + b * rfg + c
                pred_alpha = tf.nn.sigmoid(sum_a, name='predicted_alpha')
                fg_seg = tf.multiply(pred_alpha, self.imgs, name='predicted_fg')

                self.end_points['pred_alpha'] = pred_alpha
                self.end_points['fg_seg'] = fg_seg

                tf.summary.image(name='pred_alpha', tensor=pred_alpha, max_outputs=1)
                tf.summary.image(name='fg_seg', tensor=fg_seg, max_outputs=1)


    def UpConv(self, x1, x2, out_ch, id):
        shape = x2.get_shape().as_list()[1:3]
        up_x1 = tf.image.resize_bilinear(x1, shape, name='upscale_%d'%id)
        concat = tf.concat([up_x1, x2], axis=3, name='concat_%d'%id)
        out = self.InvertedBottleneck(concat, kernel_size=[3, 3], up_sample=0.25, out_ch=out_ch, stride=1,
                                    scope='up_conv_%d'%id)
        return out


    def InvertedBottleneck(self, input_tensor, kernel_size, up_sample, out_ch, stride, scope):
        with tf.variable_scope(scope):
            in_ch = input_tensor.get_shape().as_list()[-1]
            inner_size = int(up_sample * in_ch)
            x = input_tensor
            if inner_size > in_ch:
                x = slim.conv2d(x, inner_size, [1, 1], stride=1, activation_fn=tf.nn.relu6, scope='expand')
                self.end_points[scope + '/expand'] = x
            # x = slim.conv2d(x, inner_size, [1, 1], stride=1, activation_fn=tf.nn.relu6, scope='expand')
            # self.end_points[scope + '/expand'] = x

            x = slim.separable_conv2d(x, num_outputs=None,
                                      kernel_size=kernel_size,
                                      depth_multiplier=self.depth_multiplier,
                                      stride=stride,
                                      activation_fn=tf.nn.relu6,
                                      scope='depthwise')
            self.end_points[scope + '/depthwise'] = x

            out_tensor = slim.conv2d(x, out_ch, [1, 1], stride=1, activation_fn=tf.identity, scope='project')
            self.end_points[scope + '/project'] = out_tensor

            if stride == 1 and in_ch == out_ch:
                out_tensor += input_tensor
                out_tensor = tf.identity(out_tensor, name='output')

        return out_tensor


def training_scope(is_training=True,
                   weight_decay=0.00004,
                   stddev=0.09,
                   dropout_keep_prob=0.8):
    """Defines Mobilenet training scope.
    Usage:
        with tf.contrib.slim.arg_scope(mobilenet.training_scope()):
        logits, endpoints = mobilenet_v2.mobilenet(input_tensor)
        # the network created will be trainble with dropout/batch norm
        # initialized appropriately.
    Args:
        is_training: if set to False this will ensure that all customizations are
            set to non-training mode. This might be helpful for code that is reused
            across both training/evaluation, but most of the time training_scope with
            value False is not needed. If this is set to None, the parameters is not
            added to the batch_norm arg_scope.
        weight_decay: The weight decay to use for regularizing the model.
        stddev: Standard deviation for initialization, if negative uses xavier.
        dropout_keep_prob: dropout keep probability (not set if equals to None).
        bn_decay: decay for the batch norm moving averages (not set if equals to None).
    Returns:
        An argument scope to use via arg_scope.
    """
    # Note: do not introduce parameters that would change the inference
    # model here (for example whether to use bias), modify conv_def instead.
    batch_norm_params = {
        'decay': 0.997,
        'is_training': is_training,
        'trainable': is_training
    }
    if stddev < 0:
        weight_intitializer = slim.initializers.xavier_initializer()
    else:
        weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected, slim.separable_conv2d],
        weights_initializer=weight_intitializer,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params,
        trainable=is_training,
        padding='SAME'):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as sc:
                    return sc

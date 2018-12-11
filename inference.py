import os
import matplotlib.pyplot as plt
import random
from timeit import default_timer as timer
import numpy as np
import scipy.misc as sm

import tensorflow as tf
import libs.nets.MnasUnet as network
import libs.datasets.datapipe as datasets
import libs.configs.config as cfg


if __name__ == '__main__':
    tf_image = tf.placeholder(dtype=tf.float32, shape=[None, None, 3], name='inputs')
    norm_image = datasets._preprocess_for_test(tf_image)
    norm_image = tf.expand_dims(norm_image, 0)

    model = network.MnasUnet(norm_image, is_training=False)
    pred_alpha = model.end_points['pred_alpha']
    fg_seg = model.end_points['fg_seg']
    print(pred_alpha, fg_seg)
    print(model.end_points['input'])

    """ set saver for saving final model and backbone model for restore """
    saver = tf.train.Saver(max_to_keep=3)

    """ Set Gpu Env """
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    # Performing post-processing on CPU: loop-intensive, usually more efficient.
    with tf.Session() as sess:
        sess.run(init_op)
        ckpt_path = 'output/models/MnasUnet_Matting_final.ckpt'
        saver.restore(sess, ckpt_path)

        # ======================================================================
        images = os.listdir('data/test')

        ori_images = []
        alpha_imgs = []
        fg_imgs = []

        for fn in images:
            name = fn.split('.')[0]
            image = sm.imread('data/test/'+fn)
            print(image.shape)
            feed_dict = {tf_image: image}
            result_alpha, result_fg = sess.run([pred_alpha, fg_seg], feed_dict=feed_dict)
            result_alpha = result_alpha[0][:, :, 0]
            result_fg = result_fg[0]
            print(result_alpha.shape, result_fg.shape)
            h, w = image.shape[:2]

            alpha = sm.imresize(result_alpha, (h, w))
            fg = sm.imresize(result_fg, (h, w))
            print(alpha.shape, fg.shape)

            # sm.imsave('data/aa/'+name+'_alpha.jpg', alpha)
            # sm.imsave('data/aa/'+name+'_fg.jpg', fg)
            ori_images.append(image)
            alpha_imgs.append(alpha)
            fg_imgs.append(fg)

        sess.close()

    fig, axes = plt.subplots(8, 3, figsize=(15, 15))
    print(axes.shape, len(axes))
    for i in range(len(axes)):
        print(ori_images[i].shape)
        axes[i, 0].imshow(ori_images[i])
        axes[i, 0].set_xlabel("original image")

        axes[i, 1].imshow(alpha_imgs[i])
        axes[i, 1].set_xlabel("alpha_map image")

        axes[i, 2].imshow(fg_imgs[i])
        axes[i, 2].set_xlabel("foreground image")

    plt.subplots_adjust(left=0, wspace=0, hspace=1)
    plt.show()






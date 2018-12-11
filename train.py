"""
created by TJ.Park
data : 22. Nov. 2017
"""

import os, time
from time import gmtime, strftime
import numpy as np
import tensorflow as tf
import libs.nets.MnasUnet as network
import libs.nets.losses as losses
import libs.datasets.datapipe as datasets
import libs.configs.config as cfg

FLAGS = tf.app.flags.FLAGS

def _get_learning_rate(num_sample_per_epoch, global_step):
    decay_step = int((num_sample_per_epoch / FLAGS.batch_size) * FLAGS.num_epochs_per_decay)
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_step,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')

def _get_restore_vars(scope):
    scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    variables_to_restore = []
    for var in scope_vars:
        s = var.op.name
        if s.find("Momentum") != -1:
            continue
        elif s.find("moving_mean") != -1:
            continue
        elif s.find("moving_variance") != -1:
            continue
        variables_to_restore.append(var)

    return variables_to_restore


def train():
    inputs, gt_alphas = datasets.get_dataset()
    model = network.MnasUnet(inputs, is_training=True)

    total_loss = losses.compute_loss(model.end_points, gt_alphas, mode=FLAGS.mode)

    """ set the update operations for training """
    update_ops = []
    variables_to_train = tf.trainable_variables()

    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = _get_learning_rate(FLAGS.num_images, global_step)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    update_opt = optimizer.minimize(total_loss, global_step, variables_to_train)
    update_ops.append(update_opt)

    update_bns = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if len(update_bns):
        update_bn = tf.group(*update_bns)
        update_ops.append(update_bn)
    update_op = tf.group(*update_ops)

    """ set Summary and log info """
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('model_loss', model.end_points['model_loss'])
    tf.summary.scalar('regular_loss', model.end_points['regular_loss'])

    summary_op = tf.summary.merge_all()
    logdir = os.path.join(FLAGS.summaries_dir, strftime('%Y%m%d%H%M%S', gmtime()))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    summary_writer = tf.summary.FileWriter(logdir, graph=tf.Session().graph)

    """ set saver for saving final model and backbone model for restore """
    saver = tf.train.Saver(max_to_keep=3)

    """ Set Gpu Env """
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opt)) as sess:
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(FLAGS.training_checkpoint_model)
        """ resotre checkpoint of Backbone network """
        if ckpt is not None:
            lastest_ckpt = tf.train.latest_checkpoint(FLAGS.training_checkpoint_model)
            print('lastest', lastest_ckpt)
            re_saver = tf.train.Saver(var_list=tf.global_variables())
            re_saver.restore(sess, lastest_ckpt)
        else:
            restore_vars =_get_restore_vars("MnasNet")
            re_saver = tf.train.Saver(var_list=restore_vars)
            re_saver.restore(sess, "data/pretrained_models/MnasNet_224_final.ckpt")

        """ Generate threads """
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                s_time = time.time()
                _, loss, current_step = sess.run([update_op, total_loss, global_step])

                duration_time = time.time() - s_time
                print ("""iter %d: time:%.3f(sec), total-loss %.4f""" % (current_step, duration_time, loss))

                if np.isnan(loss) or np.isinf(loss):
                    print('isnan or isinf', loss)
                    raise

                if current_step % 10 == 0:
                    # write summary
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary, current_step)
                    summary_writer.flush()

                if current_step % 50 == 0:
                    # Save a checkpoint
                    save_path = 'output/training/MnasUnet_Matting.ckpt'
                    saver.save(sess, save_path, global_step=current_step)

                if current_step + 1 == FLAGS.max_iters:
                    print('max iter : %d, current_step : %d' % (FLAGS.max_iters, current_step))
                    break

        except tf.errors.OutOfRangeError:
            print('Error occured')
        finally:
            saver.save(sess, './output/models/MnasUnet_Matting_final.ckpt', write_meta_graph=False)
            coord.request_stop()

        coord.join(threads)
        sess.close()


if __name__ == "__main__":
    train()
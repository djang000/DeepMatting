import tensorflow as tf



def get_Segloss(end_points, gt_alpha):
    score_map = end_points['score_map']

    # alpha = tf.cast(tf.greater(gt_alpha, 0.3), dtype=tf.int32)
    # alpha = tf.squeeze(alpha, axis=3)
    # print(alpha.shape)
    # alpha_ohe = tf.one_hot(alpha, 2, axis=-1)
    # print(alpha_ohe.shape, score_map.shape)
    #
    # bin_mask_loss = tf.nn.softmax_cross_entropy_with_logits(labels=alpha_ohe, logits=score_map)
    # bin_mask_loss = tf.reduce_mean(bin_mask_loss)

    mask_one_hot = tf.one_hot(gt_alpha, 2, on_value=1.0, off_value=0.0, dtype=tf.float32)
    one_hot_b, one_hot_f = tf.split(mask_one_hot, num_or_size_splits=2, axis=3)
    print('one_b', one_hot_b.shape)

    tf.summary.image(name='one_hot_b', tensor=one_hot_b, max_outputs=1)
    tf.summary.image(name='one_hot_f', tensor=one_hot_f, max_outputs=1)

    bin_mask_loss = tf.nn.softmax_cross_entropy_with_logits(labels=mask_one_hot, logits=score_map)
    bin_mask_loss = tf.reduce_mean(bin_mask_loss)

    return bin_mask_loss

def get_Feather_loss(end_points, gt_alpha):
    print('======================')
    pred_alpha = end_points['pred_alpha']
    pred_fg = end_points['fg_seg']
    # gt_alpha = tf.expand_dims(gt_alpha, axis=3)
    gt_fg = gt_alpha * end_points['input']
    eps = 1e-12

    # diff_alpha = gt_alpha - pred_alpha
    #
    # alpha_losses = tf.sqrt(tf.square(diff_alpha) + eps)
    # alpha_loss = tf.reduce_mean(alpha_losses)
    # print(alpha_losses, alpha_loss)
    # tf.add_to_collection('alpha_loss', alpha_loss)
    #
    # diff_fg = gt_fg - pred_fg
    # color_losses = tf.sqrt(tf.square(diff_fg) + eps)
    # color_loss = tf.reduce_mean(color_losses)
    # print(color_losses, color_loss)
    # tf.add_to_collection('color_loss', color_loss)


    alpha_loss = tf.losses.mean_squared_error(labels=gt_alpha, predictions=pred_alpha, scope='alpha_loss')
    color_loss = tf.losses.mean_squared_error(labels=gt_fg, predictions=pred_fg, scope='color_loss')

    total_loss = alpha_loss + color_loss
    tf.add_to_collection('total_loss', total_loss)
    return total_loss


def compute_loss(end_points, gt_alpha, mode):
    """

    :param end_points: feature data
    :param gt_alpha: gt_annotation data
    :param mode: loss function mode:
                mode 1 : coarse segmentation loss,
                mode 2 : feather loss
                mode 0 : Total loss (coarse segmentation loss + feather loss)
    :return: total loss
    """
    model_loss = 0
    if mode == 1:
        print('seg_loss')
        model_loss = model_loss + get_Segloss(end_points, gt_alpha)
    elif mode == 2:
        print('feather_loss')
        model_loss = model_loss + get_Feather_loss(end_points, gt_alpha)
    else:
        print('all_loss')
        seg_loss = get_Segloss(end_points, gt_alpha)
        feather_loss = get_Feather_loss(end_points, gt_alpha)
        model_loss = seg_loss + feather_loss

    regular_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regular_loss = tf.add_n(regular_loss)
    print(regular_loss)
    print(model_loss)

    total_loss = model_loss + regular_loss
    end_points['model_loss'] = model_loss
    end_points['regular_loss'] = regular_loss
    end_points['total_loss'] = total_loss

    return total_loss
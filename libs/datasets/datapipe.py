import os, glob, math
import numpy as np
from PIL import Image
import scipy.io, scipy.misc

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'gt_alpha': 'An annotation image with varying size. (pixel-level masks)',
}


def _flip_image(image):
    return tf.reverse(image, axis=[1])

def _flip_gt_masks(gt_masks):
    return tf.reverse(gt_masks, axis=[1])

def _rotate_images(image, gt_mask, seed):
    ratio = seed * 2.0 - 1.0
    angle = (ratio*30.0) * (math.pi / 180.0)
    rotate_image = tf.contrib.image.rotate(image, angle)
    rotate_mask = tf.contrib.image.rotate(gt_mask, angle)
    return rotate_image, rotate_mask


def _preprocess_for_training(image, gt_mask):
    """ step 1. random resize """
    rand = np.random.rand(4)

    """ step 1. random cropping """
    if rand[3] > 0.5:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32,
                           shape=[1, 1, 4])
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(0.75, 1.33),
            area_range=(0.7, 1.0),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        image = tf.slice(image, bbox_begin, bbox_size)
        gt_mask = tf.slice(gt_mask, bbox_begin, bbox_size)
        image.set_shape([None, None, 3])

    """ step 2. resize """
    # ratio = np.random.uniform(0.8, 1.2)
    # new_width = int(300.0 * ratio)
    # new_height = int(400.0 * ratio)
    # print(new_width, new_height)

    new_width = 400
    new_height = 400

    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
    image = tf.squeeze(image, axis=[0])

    gt_mask = tf.expand_dims(gt_mask, 0)
    gt_mask = tf.cast(gt_mask, tf.float32)
    gt_mask = tf.image.resize_bilinear(gt_mask, [new_height, new_width], align_corners=False)
    gt_mask = tf.squeeze(gt_mask, axis=[0])

    image = tf.reshape(image, [new_height, new_width, 3])
    gt_mask = tf.reshape(gt_mask, [new_height, new_width, 1])

    """ step 3. random flipping """
    flip_thresh = tf.constant(rand[1], dtype=tf.float32)
    val = tf.constant(0.5, dtype=tf.float32)
    image, gt_mask = tf.cond(tf.greater_equal(flip_thresh, val),
                              lambda : (_flip_image(image),
                                        _flip_gt_masks(gt_mask)),
                              lambda : (image, gt_mask))


    """ step 4. random rotation """
    rotate_thresh = tf.constant(rand[2], dtype=tf.float32)
    image, gt_mask = tf.cond(tf.greater_equal(rotate_thresh, val),
                              lambda : (_rotate_images(image, gt_mask, rand[3])),
                              lambda : (image, gt_mask))


    """ step 5. convert [0, 255] to [0.0, 1.0] """
    image = tf.image.convert_image_dtype(image, tf.float32)
    gt_mask = tf.image.convert_image_dtype(gt_mask, tf.float32)
    image = image / 255.0
    gt_mask = gt_mask / 255.0

    return image, gt_mask

def _preprocess_for_test(image):
    new_width = 400
    new_height = 400

    """ step 1. min size resize """
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
    image = tf.squeeze(image, axis=[0])

    """ step 2. convert [0, 255] to [0.0, 1.0] """
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.cast(image, tf.float32)
    norm_image = image / 255.0
    print(norm_image)
    return norm_image

def get_dataset():
    # image_files = sorted([os.path.join('data/coco/images', file) for file in os.listdir('data/person_mask')])
    # mask_files = sorted([os.path.join('data/person_mask', file) for file in os.listdir('data/person_mask')])
    image_files = sorted([os.path.join('data/images_data', file) for file in os.listdir('data/images_data')])
    mask_files = sorted([os.path.join('data/mask_data', file) for file in os.listdir('data/mask_data')])
    print(len(image_files), len(mask_files))

    # print(mask_files[0])
    # im = scipy.misc.imread(mask_files[0])
    # print(im.shape)
    # scipy.misc.imsave('alpha.jpg', im)
    # ssss
    # for i in range(5):
    #     print(image_files[i])
    #     im = Image.open(image_files[i])
    #     m = Image.open(mask_files[i])
    #     im.save('new%d.jpg'%i)
    #     m.save('m%d.jpg'%i)
    images = tf.convert_to_tensor(image_files)
    masks = tf.convert_to_tensor(mask_files)
    input_queue = tf.train.slice_input_producer([images, masks])
    image = tf.read_file(input_queue[0])
    mask = tf.read_file(input_queue[1])

    image = tf.image.decode_image(image, channels=3)
    gt_mask = tf.image.decode_image(mask, channels=1)

    # image = tf.reshape(image, [800, 600, 3])
    # gt_mask = tf.reshape(gt_mask, [800, 600, 1])

    # input_queue = tf.train.string_input_producer(tf.train.match_filenames_once(image_files), capacity=200)
    # mask_queue = tf.train.string_input_producer(tf.train.match_filenames_once(mask_files), capacity=200)
    # image_reader = tf.WholeFileReader()
    #
    # _, images = image_reader.read(input_queue)
    # _, masks = image_reader.read(mask_queue)
    #
    # imageA = tf.image.decode_jpeg(images)
    # imageB = tf.image.decode_jpeg(masks)
    # image = tf.cast(tf.reshape(imageA, shape=[800, 600, 3]), dtype=tf.float32)
    # gt_mask = tf.cast(tf.reshape(imageB, shape=[800, 600, 1]), dtype=tf.float32)

    preprocessed_image, preprocessed_gt_masks = _preprocess_for_training(image, gt_mask)
    batchs = tf.train.shuffle_batch([preprocessed_image, preprocessed_gt_masks],
                                      batch_size=FLAGS.batch_size,
                                      num_threads=4,
                                      capacity=512,
                                      min_after_dequeue=300)

    batch_images, batch_masks = batchs[0], batchs[1]


    tf.summary.image(name='input', tensor=batch_images, max_outputs=1)
    tf.summary.image(name='gt_mask', tensor=batch_masks, max_outputs=1)

    return batch_images, batch_masks #batchs[0], batchs[1]



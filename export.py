import os
import numpy as np
import tensorflow as tf
import libs.datasets.datapipe as datasets
import libs.nets.MnasUnet as network
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python import pywrap_tensorflow

if __name__ == '__main__':
    INPUT_POINT = "inputs"
    PB_ALPHA = "feathering/predicted_alpha"
    PB_FG = "feathering/predicted_fg"
    OUTPUT_PB_FILENAME = "output/models/DeepMatting_model.pb"


    # checkpoint_path = "output/models/MnasUnet_Matting_final.ckpt"
    #
    # reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    # var_to_shape_map = reader.get_variable_to_shape_map()
    #
    # for key in var_to_shape_map:
    #     print("tensor_name: ", key)
    #     # print(reader.get_tensor(key))  # Remove this is you want to print only variable names
    #
    #
    # sssss

    graph = tf.Graph()
    with graph.as_default():
        tf_image = tf.placeholder(dtype=tf.float32, shape=[None, 400, 400, 3], name='inputs')
        # norm_image = datasets._preprocess_for_test(tf_image)
        # norm_image = tf.expand_dims(norm_image, 0)
        # print(norm_image)

        model = network.MnasUnet(tf_image, is_training=False)

        # Training model
        saver = tf.train.Saver()
        with tf.Session(graph=graph) as sess:
            save_path = "output/models/MnasUnet_Matting_final.ckpt"
            saver.restore(sess, save_path)

            constant_graph = convert_variables_to_constants(sess,
                                                            sess.graph_def,
                                                            [INPUT_POINT, PB_ALPHA, PB_FG])

            optimized_constant_graph = optimize_for_inference(constant_graph,
                                                              [INPUT_POINT],
                                                              [PB_ALPHA, PB_FG],
                                                              tf.float32.as_datatype_enum)

            # Generate PB file and we also generate text file for debug on graph
            tf.train.write_graph(optimized_constant_graph, '.', OUTPUT_PB_FILENAME, as_text=False)
            tf.train.write_graph(optimized_constant_graph, '.', OUTPUT_PB_FILENAME + ".txt", as_text=True)

        # print PB file size
        filesize = os.path.getsize(OUTPUT_PB_FILENAME)
        filesize_mb = filesize / 1024 / 1024
        print(str(round(filesize_mb, 3)) + " MB")

        sess.close()

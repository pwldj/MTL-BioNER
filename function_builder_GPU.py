"""doc."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import tensorflow as tf
import modeling
import xlnet


def get_crf_outputs(FLAGS, features, is_training):
    """Loss for downstream span-extraction QA tasks such as SQuAD."""

    inp = tf.transpose(features["input_ids"], [1, 0])
    seg_id = tf.transpose(features["segment_ids"], [1, 0])
    inp_mask = tf.transpose(features["input_mask"], [1, 0])

    if FLAGS.label_mode == "normal":
        label = features["label_ids"]
    elif FLAGS.label_mode == "X":
        label = features["label_x_id"]
    elif FLAGS.label_mode == "gather":
        label = features["label_gather"]
    else:
        raise ValueError("unsupport label mode {}".format(FLAGS.label_mode))

    if FLAGS.label_mask == "normal":
        mask = 1 - features["input_mask"]
        re_mask = features["label_mask_x"]
    elif FLAGS.label_mask == "X":
        mask = features["label_mask_x"]
        re_mask = features["label_mask_x"]
    elif FLAGS.label_mask == "gather":
        mask = features["label_mask_gather"]
        re_mask = features["label_mask_gather"]

    else:
        raise ValueError("unsupport mask mode {}".format(FLAGS.label_mode))

    max_seq_length = FLAGS.max_seq_length
    batch_size = tf.shape(inp)[1]
    if FLAGS.label_mode == "X":
        classes = FLAGS.crf_classes + 1
    else:
        classes = FLAGS.crf_classes

    xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    run_config = xlnet.create_run_config(is_training, True, FLAGS)

    xlnet_model = xlnet.XLNetModel(
        xlnet_config=xlnet_config,
        run_config=run_config,
        input_ids=inp,
        seg_ids=seg_id,
        input_mask=inp_mask)
    output = xlnet_model.get_sequence_output()
    initializer = xlnet_model.get_initializer()

    with tf.variable_scope("crf_layer"):
        output = tf.transpose(output, [1, 0, 2])
        start_logits = tf.layers.dense(
            output,
            classes,
            kernel_initializer=initializer)

        if FLAGS.label_mode == "gather":
            flat_offsets = tf.reshape(
                tf.range(0, batch_size, dtype=tf.int32) * max_seq_length, [-1, 1])
            flat_positions = tf.reshape(features["label_index"] + flat_offsets, [-1])
            flat_sequence_tensor = tf.reshape(start_logits, [batch_size * max_seq_length, classes])
            start_logits = tf.gather(flat_sequence_tensor, flat_positions)
            start_logits = tf.reshape(start_logits, [batch_size, max_seq_length, classes])

        if FLAGS.no_crf:
            logits = tf.nn.log_softmax(start_logits)
            one_hot_target = tf.one_hot(label, classes)
            per_example_loss = -tf.reduce_sum(logits * one_hot_target, -1)
            numerator = tf.reduce_sum(tf.reshape(mask * per_example_loss, [-1]))
            denominator = tf.reduce_sum(tf.reshape(mask, [-1])) + 1e-5
            total_loss = numerator / denominator
            logits = tf.argmax(logits, axis=-1)
        else:
            seq_len = tf.reduce_sum(tf.cast(mask, tf.int64), axis=-1)
            transition_params = tf.get_variable('trans', [classes, classes], dtype=tf.float32,
                                                initializer=tf.zeros_initializer)
            per_example_loss, transition_params = tf.contrib.crf.crf_log_likelihood(start_logits, label, seq_len,
                                                                                    transition_params=transition_params)
            logits, tf_score = tf.contrib.crf.crf_decode(start_logits, transition_params, seq_len)
            per_example_loss = -per_example_loss
            total_loss = tf.reduce_mean(per_example_loss)

    return total_loss, per_example_loss, logits, label, re_mask


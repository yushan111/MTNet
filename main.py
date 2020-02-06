from models.MTNet_keras_1 import MTNet
# from models.MTNet import *
from models.MTNet_keras import MTNetKeras
from configs.config import *
from preprocess.get_data import *

from functools import reduce
from operator import mul
import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

# GPU setting
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
SCORE_TYPES = [['MAE', 'RMSE'], ['CORR', 'RSE']]


CONFIG = BJpmConfig
DS_HANDLER = BJPMDataset
score_type_index = 0

# CONFIG = TaxiNYConfig
# DS_HANDLER = TaxiNYDataset
# score_type_index = 1


is_train = True

MODEL_DIR = os.path.join('logs', 'checkpoints')
LOG_DIR = os.path.join('logs', 'graphs')

def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

def make_config_string(config):
    return "T%s_W%s_n%s_hw%s_dropin%s_cnn%s_rnn%s" % \
           (config.T, config.W, config.n, config.highway_window, config.input_keep_prob,
            config.en_conv_hidden_size, config.en_rnn_hidden_sizes)

def make_log_dir(config, ds_handler):
    return os.path.join(LOG_DIR, ds_handler.name, 'horizon' + str(config.horizon),make_config_string(config))

def make_model_path(config, ds_handler):
    dir = os.path.join(MODEL_DIR, ds_handler.name, 'horizon' + str(config.horizon))
    if not os.path.exists(dir):
        os.makedirs(dir)
    return os.path.join(dir, make_config_string(config), 'mtnet.ckpt')

def calc_rse(y_real_list, y_pred_list):
    rse_numerator = np.sum(np.subtract(y_pred_list, y_real_list) ** 2)
    rse_denominator = np.sum(np.subtract(y_real_list, np.mean(y_real_list)) ** 2)
    rse = np.sqrt(np.divide(rse_numerator, rse_denominator))
    return rse

def calc_corr(y_real_list, y_pred_list):
    y_real_mean = np.mean(y_real_list, axis = 0)
    y_pred_mean = np.mean(y_pred_list, axis = 0)

    numerator = np.sum((y_real_list - y_real_mean) * (y_pred_list - y_pred_mean), axis = 0)
    denominator_real = np.sqrt(np.sum((y_real_list - y_real_mean) ** 2, axis = 0))
    denominator_pred = np.sqrt(np.sum((y_pred_list - y_pred_mean) ** 2, axis = 0))
    denominator = denominator_real * denominator_pred
    corr = np.mean(numerator / denominator)

    return corr

def run_one_epoch(sess, model, batch_data, summary_writer, ds_handler, epoch_num, is_train = True):
    # reset statistics variables
    sess.run(model.reset_statistics_vars)

    if is_train :
        run_func = model.train
    else:
        run_func = model.predict

    y_pred_list = []
    y_real_list = []
    loss_list = []

    # step = 0
    # loss = 0.10731270909309387
    # loss = 0.1073080450296402
    # predict = 0.3857574462890625
    # loss_new, pred_new = run_func(batch_data[step], sess)
    # assert np.equal(loss, loss_new), "new_loss is {}, which is not equal to {}".format(loss_new, loss)
    # assert np.equal(predict, pred_new[0][0]), "new_pred is {}, which is not equal to {}".format(pred_new, predict)
    # exit(0)

    for ds in batch_data:
        loss, pred = run_func(ds, sess)

        y_pred_list.append(pred)
        y_real_list.append(ds[-1])
        loss_list.append(loss)

    # inverse norm
    y_pred_list = np.reshape(y_pred_list, [-1, model.config.K])
    y_real_list = np.reshape(y_real_list, [-1, model.config.K])

    y_pred_list = ds_handler.inverse_transform(y_pred_list)
    y_real_list = ds_handler.inverse_transform(y_real_list)

    # real value loss
    loss = np.mean(loss_list)

    # summary
    # model's summary
    summary = sess.run(model.merged_summary)
    summary_writer.add_summary(summary, epoch_num)
    # other summary
    if score_type_index == 0:
        mae = np.mean(abs(np.subtract(y_pred_list, y_real_list)))
        rmse = np.sqrt(np.mean(np.subtract(y_pred_list, y_real_list) ** 2))

        real_mae_summary = tf.Summary()
        real_mae_summary.value.add(tag='real_mae', simple_value=mae)
        summary_writer.add_summary(real_mae_summary, epoch_num)

        real_rmse_summary = tf.Summary()
        real_rmse_summary.value.add(tag='real_rmse', simple_value=rmse)
        summary_writer.add_summary(real_rmse_summary, epoch_num)
        return loss, mae, rmse
    else:
        rse = calc_rse(y_real_list, y_pred_list)
        corr = calc_corr(y_real_list, y_pred_list)

        real_rse_summary = tf.Summary()
        real_rse_summary.value.add(tag='real_rse', simple_value=rse)
        summary_writer.add_summary(real_rse_summary, epoch_num)

        real_corr_summary = tf.Summary()
        real_corr_summary.value.add(tag='real_corr', simple_value=corr)
        summary_writer.add_summary(real_corr_summary, epoch_num)
        return loss, corr, rse


def evaluate(model, val_data, ds_handler, score_type_index):
    val_x = val_data[0]
    y_real = val_data[1]
    # val_x = []
    # val_y = []
    # val_q = []
    # for ds in val_batch_data:
    #     val_x.append(ds[0])
    #     val_q.append(ds[1])
    #     val_y.append(ds[2])
    #     # val_y.append(ds[1])
    # val_x = np.concatenate(val_x), np.concatenate(val_q)
    # y_real = np.concatenate(val_y)

    y_pred = model.predict(val_x)
    # # inverse norm
    # print("the shape of y_pred is", y_pred.shape)
    # print("the shape of y_real is", y_real.shape)

    y_pred = np.reshape(y_pred, [-1, model.config.K])
    y_real = np.reshape(y_real, [-1, model.config.K])

    y_pred = ds_handler.inverse_transform(y_pred)
    y_real = ds_handler.inverse_transform(y_real)

    if score_type_index == 0:
        mae = np.mean(abs(np.subtract(y_pred, y_real)))
        rmse = np.sqrt(np.mean(np.subtract(y_pred, y_real) ** 2))
        return mae, rmse
    else:
        rse = calc_rse(y_real, y_pred)
        corr = calc_corr(y_real, y_pred)
        return corr, rse


def run_one_config(config):
    epochs = 50

    # learning rate decay
    max_lr = 0.001
    min_lr = 0.001
    decay_epochs = 60

    # build model
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        model = MTNet(config)
        saver = tf.train.Saver()
        # data process
        ds_handler = DS_HANDLER(config)
        train_batch_data = ds_handler.get_all_batch_data(config, 'T')
        valid_batch_data = ds_handler.get_all_batch_data(config, 'V')

        # generate log and model stored paths
        log_path = make_log_dir(config, ds_handler)
        model_path = make_model_path(config, ds_handler)

        # tensor board
        train_writer = tf.summary.FileWriter(os.path.join(log_path, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(log_path, 'test'))

        print('----------Train Config:', make_config_string(config), '. Total epochs:', epochs)
        print('Trainable parameter count:', get_num_params())

        # Actually run
        sess.run(tf.global_variables_initializer())

        best_score = float('inf')

        # indicate the score name
        score1_name, score2_name= SCORE_TYPES[score_type_index]

        for i in range(epochs):
            # decay lr
            config.lr = min_lr + (max_lr - min_lr) * math.exp(-i/decay_epochs)

            # train one epoch
            run_one_epoch(sess, model, train_batch_data, train_writer, ds_handler, i, True)
            # evaluate
            # print(i)
            if (i + 1) % 1 == 0:
                loss, scope1, score2 = run_one_epoch(sess, model, valid_batch_data, test_writer, ds_handler, i, False)
                if best_score > score2:
                    best_score = score2
                    # save model
                    # saver.save(sess, model_path)
                print('Epoch', i + 1, 'Test Loss:', loss, score1_name,':', scope1, score2_name, ':', score2)

        print('---------Best score:', score2_name, ':', best_score)

    # free default graph
    tf.reset_default_graph()


def run_keras(config):
    epochs = 50
    # learning rate decay
    max_lr = 0.001
    min_lr = 0.001
    decay_epochs = 60

    # build model
    model = MTNetKeras()
    # data process
    ds_handler = DS_HANDLER(config)
    train_batch_data = ds_handler.get_all_batch_data(config, 'T')
    valid_batch_data = ds_handler.get_all_batch_data(config, 'V')
    def get_data(batch_data):
        x = []
        q = []
        y = []
        for ds in batch_data:
            x.append(ds[0])
            q.append(ds[1])
            y.append(ds[2])
        input_x = np.concatenate(x)
        input_q = np.concatenate(q)
        input_y = np.concatenate(y)
        return input_x, input_q, input_y
    train_x, train_q, train_y = get_data(train_batch_data)
    val_x, val_q, val_y = get_data(valid_batch_data)

    # train_data = ds_handler.get_all_data(config, 'T')
    # valid_data = ds_handler.get_all_data(config, 'V')

    # generate log and model stored paths
    log_path = make_log_dir(config, ds_handler)
    model_path = make_model_path(config, ds_handler)

    print('----------Train Config:', make_config_string(config), '. Total epochs:', epochs)
    score1_name, score2_name = SCORE_TYPES[score_type_index]
    best_score = float('inf')
    for i in range(epochs//5):
        # decay lr
        config.lr = min_lr + (max_lr - min_lr) * math.exp(-i / decay_epochs)
        val_loss = model.fit_eval([train_x, train_q], train_y,
                                  validation_data=([val_x, val_q], val_y),
                                  epochs=5, config=config)
        score1, score2 = evaluate(model, ([train_x, train_q], train_y), ds_handler,
                                  score_type_index)
        if best_score > score2:
            best_score = score2
            # save model
            # model.save(model_path)
            print('Epoch', (i + 1)*5, 'Test Loss:', val_loss,
                  score1_name, ':', score1,
                  score2_name, ':', score2)

    print('---------Best score:', score2_name, ':', best_score)


if __name__ == '__main__':
    # tf.random.set_random_seed(0)
    # np.random.seed(0)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config = CONFIG()
    for en_conv_hidden_size in [32, 64]:
        config.en_conv_hidden_size = en_conv_hidden_size
        for en_rnn_hidden_sizes in [[32, 32], [32, 64]]:
            config.en_rnn_hidden_sizes = en_rnn_hidden_sizes

            # run_keras(config)
            run_one_config(config)
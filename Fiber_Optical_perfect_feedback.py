# -*- coding: utf-8 -*-
"""Fiber_Optical_v1

This file realizes alternating algorithm over fiber optical channel with M=16

The feedback link is a perfect link, with which the real per sample loss is sent to the transmitter


"""

import numpy as np
import os
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import math
import time
from matplotlib.animation import FuncAnimation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

M = 16
P_in_dBm = -5  #dBm
P_in_W = 10**(P_in_dBm / 10) / 1000  # W

lr_receiver = 0.008
lr_transmitter = 0.001

sigma_pi = np.sqrt(0.001)  # Variance for Gaussian policy

tx_layers = 3
rx_layers = 3
NN_T = 30  # Number of neurons in each hidden layer
NN_R = 50

epsilon = 0.000000001 # to avoid non value while doing logarithm

# one hot encoding
messages = np.array(np.arange(1, M+1))
one_hot_encoded = to_categorical(messages-1)
one_hot_labels = np.transpose(one_hot_encoded)

with tf.variable_scope('Transmitter'):
    WT = []
    BT = []
    for num_layer in range(1, tx_layers+1):
        w_name = 'WT' + str(num_layer)
        b_name = 'BT' + str(num_layer)
        if num_layer == 1:
            weights = tf.get_variable(w_name, [NN_T, M], dtype='float64', initializer=tf.contrib.layers.xavier_initializer(seed=1))
            bias = tf.get_variable(b_name, [NN_T, 1], dtype='float64', initializer=tf.contrib.layers.xavier_initializer(seed=1))
            WT = np.append(WT, weights)
            BT = np.append(BT, bias)

        elif num_layer == tx_layers:
            weights = tf.get_variable(w_name, [2, NN_T], dtype='float64', initializer=tf.contrib.layers.xavier_initializer(seed=1))
            bias = tf.get_variable(b_name, [2, 1], dtype='float64', initializer=tf.contrib.layers.xavier_initializer(seed=1))
            WT = np.append(WT, weights)
            BT = np.append(BT, bias)
        else:
            weights = tf.get_variable(w_name, [NN_T, NN_T], dtype='float64', initializer=tf.contrib.layers.xavier_initializer(seed=1))
            bias = tf.get_variable(b_name, [NN_T, 1], dtype='float64', initializer=tf.contrib.layers.xavier_initializer(seed=1))
            WT = np.append(WT, weights)
            BT = np.append(BT, bias)


def transmitter(in_message):
    layer = []
    for n_tx in range(1, tx_layers+1):
        if n_tx == 1:
            layer = tf.nn.relu(tf.add(tf.matmul(WT[n_tx - 1], in_message), BT[n_tx - 1]))  # input layer
        elif n_tx < tx_layers:
            layer = tf.nn.relu(tf.add(tf.matmul(WT[n_tx-1], layer), BT[n_tx-1]))  # input layer
        else:
            layer = tf.add(tf.matmul(WT[n_tx-1], layer), BT[n_tx-1])
    return layer


with tf.variable_scope('Receiver'):
    WR = []
    BR = []
    for num_layer in range(1, rx_layers + 1):
        w_name = 'WR' + str(num_layer)
        b_name = 'BR' + str(num_layer)
        if num_layer == 1:
            weights = tf.get_variable(w_name, [NN_R, 2], dtype='float64', initializer=tf.contrib.layers.xavier_initializer(seed=1))
            bias = tf.get_variable(b_name, [NN_R, 1], dtype='float64', initializer=tf.contrib.layers.xavier_initializer(seed=1))
            WR = np.append(WR, weights)
            BR = np.append(BR, bias)

        elif num_layer == rx_layers:
            weights = tf.get_variable(w_name, [M, NN_R], dtype='float64', initializer=tf.contrib.layers.xavier_initializer(seed=1))
            bias = tf.get_variable(b_name, [M, 1], dtype='float64', initializer=tf.contrib.layers.xavier_initializer(seed=1))
            WR = np.append(WR, weights)
            BR = np.append(BR, bias)
        else:
            weights = tf.get_variable(w_name, [NN_R, NN_R], dtype='float64', initializer=tf.contrib.layers.xavier_initializer(seed=1))
            bias = tf.get_variable(b_name, [NN_R, 1], dtype='float64', initializer=tf.contrib.layers.xavier_initializer(seed=1))
            WR = np.append(WR, weights)
            BR = np.append(BR, bias)


def receiver(in_symbols):
    layer = []
    for n_rx in range(1, rx_layers + 1):
        if n_rx == 1:
            layer = tf.nn.relu(tf.add(tf.matmul(WR[n_rx - 1], in_symbols), BR[n_rx - 1]))  # input layer
        elif n_rx < rx_layers:
            layer = tf.nn.relu(tf.add(tf.matmul(WR[n_rx - 1], layer), BR[n_rx - 1]))  # input layer
        else:
            layer = tf.nn.softmax(tf.add(tf.matmul(WR[n_rx - 1], layer), BR[n_rx - 1]), 0)  # output layer
    return layer


def normalization(in_message):  # normalize average energy to 1
    m = tf.size(in_message[0, :])
    square = tf.square(in_message)
    inverse_m = 1 / m
    inverse_m = tf.cast(inverse_m, tf.float64)
    E_abs = inverse_m * tf.reduce_sum(square)
    power_norm = tf.sqrt(E_abs)  # average power per message
    y = in_message / power_norm  # average power per message normalized to 1
    return y


def power_constrain(in_message):
    P_in = tf.cast(P_in_W, tf.float64)
    out_put = tf.sqrt(P_in) * in_message
    return out_put


def compute_loss(prob_distribution, labels):
    loss = -tf.reduce_mean(tf.reduce_sum(tf.log(prob_distribution + epsilon) * labels, 0))
    return loss


def perturbation(input_signal):
    rows = tf.shape(input_signal)[0]
    columns = tf.shape(input_signal)[1]
    noise = tf.random_normal([rows, columns], mean=0.0, stddev=sigma_pi, dtype=tf.float64, seed=None, name=None)
    perturbed_signal = input_signal + noise  # add perturbation so as to do exploration
    return perturbed_signal


def compute_per_sample_loss(prob_distribution, labels):
    # this is actually the receiver, use the same training set as receiver, so that it knows what message is transmitted
    sample_loss = -tf.reduce_sum(tf.log(prob_distribution + epsilon) * labels, 0)
    return sample_loss


def policy_function(X_p, transmitter_output):  # problem occurs in this function
    gaussian_norm = tf.add(tf.square(X_p[0]-transmitter_output[0]), tf.square(X_p[1]-transmitter_output[1]))
    sigma_pi_square = tf.cast(tf.square(sigma_pi), 'float64')
    pi_theta = tf.multiply(tf.divide(1, np.multiply(np.pi, sigma_pi_square)),
                           tf.exp(-tf.divide(gaussian_norm, sigma_pi_square)))
    return pi_theta


def symbol_error_rate(in_snr):
    k_norm = np.sqrt(3 / (2 * (M - 1)))
    in_snr_linear = np.power(10, in_snr/10)
    sym_error_rate = 2 * (1 - 1 / np.sqrt(M)) * math.erfc(k_norm * np.sqrt(in_snr_linear)) - (
                1 - 2 / np.sqrt(M) + 1 / M) * np.square(math.erfc(k * np.sqrt(in_snr_linear)))
    return sym_error_rate


# Parameters for fiber channel:
gamma = 1.27  # non-linearity parameter
L = 2000  # total link length
K = 20  #
P_noise_dBm = -21.3  # dBw
P_noise_W = 10**(P_noise_dBm / 10) / 1000
sigma = np.sqrt(P_noise_W / K) / np.sqrt(2)


def fiber_channel(noise_variance, channel_input):
    num_inputs = tf.shape(channel_input)[1]
    channel_output = channel_input
    sigma_n = tf.cast(noise_variance, tf.float64)
    for k in range(1, K+1):
        xr = channel_output[0, :]
        xi = channel_output[1, :]
        xr = tf.reshape(xr, [1, num_inputs])
        xi = tf.reshape(xi, [1, num_inputs])
        theta0 = gamma * L * (xr ** 2 + xi ** 2) / K
        theta = tf.cast(theta0, tf.float64)
        r1 = xr * tf.cos(theta) - xi * tf.sin(theta)
        r2 = xr * tf.sin(theta) + xi * tf.cos(theta)
        r = tf.concat([r1, r2], 0)
        noise = tf.random_normal([2, num_inputs], mean=0.0, stddev=sigma_n, dtype=tf.float64)
        channel_output = r + noise
    return channel_output



MESSAGES = tf.placeholder('float64', [M, None])
LABELS = tf.placeholder('float64', [M, None])

# Train receiver:
encoded_signals = transmitter(MESSAGES)
normalized_signals = normalization(encoded_signals)

R_power_cons_signals = power_constrain(normalized_signals)
R_received_signals = fiber_channel(sigma, R_power_cons_signals)

RECEIVED_SIGNALS = tf.placeholder('float64', [2, None])
R_probability_distribution = receiver(RECEIVED_SIGNALS)
cross_entropy = compute_loss(R_probability_distribution, LABELS)
Rec_Var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Receiver')
receiver_optimizer = tf.train.AdamOptimizer(learning_rate=lr_receiver).minimize(cross_entropy, var_list=Rec_Var_list)

# Train Transmitter
perturbed_signals = perturbation(normalized_signals)  # action taken by the agent (transmitter)
PERTURBED_SIGNALS = tf.placeholder('float64', [2, None])

T_power_cons_signals = power_constrain(PERTURBED_SIGNALS)
T_received_signals = fiber_channel(sigma, T_power_cons_signals)

T_probability_distribution = receiver(T_received_signals)
per_sample_loss = compute_per_sample_loss(T_probability_distribution, LABELS)  # constant per_sample_loss
SAMPLE_LOSS = tf.placeholder('float64', [1, None])

policy = policy_function(PERTURBED_SIGNALS, normalized_signals)
reward_function = tf.reduce_mean(tf.multiply(SAMPLE_LOSS, tf.log(policy)))
Tran_Var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Transmitter')
transmitter_optimizer = tf.train.AdamOptimizer(learning_rate=lr_transmitter).minimize(reward_function, var_list=Tran_Var_list)


saver = tf.train.Saver()
save_dir = 'FIBER_NN_parameters'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')


Main_loops = 4000
batch_R = 64
batch_T = 64
rec_loops = 30
tran_loops = 20
start_time = time.time()
cons_points = np.empty([1, 2, M])  # create an empty array to hold all the constellation points

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('M=', M)
    print('Input power: ', P_in_dBm, ' dBm')
    print('Noise power: ', P_noise_dBm, 'dBm')

    loss_func = []
    reward_func = []
    for loop in range(0, Main_loops):
        if loop % 500 == 0:
            print('num of iterations=', loop)

        train_samples = np.copy(one_hot_labels)
        train_samples = np.tile(train_samples, rec_loops * batch_R)
        rec_sig = sess.run(R_received_signals,
                           feed_dict={MESSAGES: train_samples})  # constant samples to train receiver
        for train_receiver_iteration in range(0, rec_loops):
            indexes = np.arange(train_receiver_iteration * batch_R * M,
                                (train_receiver_iteration + 1) * batch_R * M)
            label_batch = np.copy(train_samples[:, indexes])
            message_batch = np.copy(rec_sig[:, indexes])
            Cross_entropy, _ = sess.run([cross_entropy, receiver_optimizer],
                                        feed_dict={RECEIVED_SIGNALS: message_batch, LABELS: label_batch})

        for train_transmitter_iteration in range(0, tran_loops):
            label_batch = np.copy(one_hot_labels)
            label_batch = np.tile(label_batch, batch_T)

            perturbed_sig = sess.run(perturbed_signals, feed_dict={MESSAGES: label_batch})  # action is constant
            sample_loss_constant = sess.run(per_sample_loss,
                                            feed_dict={PERTURBED_SIGNALS: perturbed_sig, LABELS: label_batch})
            sample_loss_constant.shape = [1, sample_loss_constant.size]
            Reward_function, _ = sess.run([reward_function, transmitter_optimizer],
                                          feed_dict={MESSAGES: label_batch,
                                                     PERTURBED_SIGNALS: perturbed_sig,
                                                     SAMPLE_LOSS: sample_loss_constant})

        if loop % 10 == 0:
            transmitted_signal = sess.run([R_power_cons_signals],
                                          feed_dict={MESSAGES: one_hot_labels})  # action is constant
            new_points = np.asarray(transmitted_signal)
            cons_points = np.concatenate([cons_points, new_points], axis=0)



        # run some more iterations and increase batchsize, so as to avoid variacne introduced by mini-batch
        if loop == Main_loops-1:
            for more_training in np.array(0, 10):
                for train_transmitter_iteration in range(0, tran_loops):
                    label_batch = np.copy(one_hot_labels)
                    label_batch = np.tile(label_batch, batch_T * 100)

                    perturbed_sig = sess.run(perturbed_signals, feed_dict={MESSAGES: label_batch})  # action is constant
                    sample_loss_constant = sess.run(per_sample_loss,
                                                    feed_dict={PERTURBED_SIGNALS: perturbed_sig, LABELS: label_batch})
                    sample_loss_constant.shape = [1, sample_loss_constant.size]
                    Reward_function, _ = sess.run([reward_function, transmitter_optimizer],
                                                  feed_dict={MESSAGES: label_batch,
                                                             PERTURBED_SIGNALS: perturbed_sig,
                                                             SAMPLE_LOSS: sample_loss_constant})


                train_samples = np.copy(one_hot_labels)
                train_samples = np.tile(train_samples, rec_loops * batch_R * 100)
                rec_sig = sess.run(R_received_signals,
                                   feed_dict={MESSAGES: train_samples})  # constant samples to train receiver
                for train_receiver_iteration in range(0, rec_loops):
                    indexes = np.arange(train_receiver_iteration * batch_R * 100 * M,
                                        (train_receiver_iteration + 1) * batch_R * 100 * M)
                    label_batch = np.copy(train_samples[:, indexes])
                    message_batch = np.copy(rec_sig[:, indexes])
                    Cross_entropy, _ = sess.run([cross_entropy, receiver_optimizer],
                                                feed_dict={RECEIVED_SIGNALS: message_batch, LABELS: label_batch})

            saver.save(sess=sess, save_path=save_path)



fig, ax = pl.subplots(figsize=(5, 5))
ax.set(xlim=(-0.03, 0.03), ylim=(-0.03, 0.03))
ax.grid()
scat = ax.scatter(cons_points[0, 0, :], cons_points[0, 1, :])
def update(frame_number):
    x = cons_points[frame_number, 0, :]
    x.shape = [x.size, 1]
    y = cons_points[frame_number, 1, :]
    y.shape = [y.size, 1]
    const = np.concatenate([x, y], 1)
    scat.set_offsets(const)
    ax.set_title('iteration {}'.format(frame_number*10))


animation = FuncAnimation(fig, update, interval=200, frames=cons_points.shape[0])
print()
animation.save('moving_constellation.mp4')
pl.show()



with tf.Session() as sess:
    saver.restore(sess=sess, save_path=save_path)

    x = np.arange(-0.1, 0.1, 0.0001)
    xx, yy = np.meshgrid(x, x)
    x = xx.reshape(1, xx.size)
    y = yy.reshape(1, xx.size)
    xymesh = np.concatenate((x, y), axis=0)
    output = sess.run(R_probability_distribution, feed_dict={RECEIVED_SIGNALS: xymesh})
    z = np.argmax(output, axis=0).reshape(2000, 2000)

    label_batch = np.copy(one_hot_labels)
    num = 64
    label_batch = np.tile(label_batch, num)

    transmitted_signal, per_sig, r_rec = sess.run([R_power_cons_signals, perturbed_signals, R_received_signals],
                                                  feed_dict={MESSAGES: label_batch})  # action is constant
    power_con_sig, fiber_signal = sess.run([T_power_cons_signals, T_received_signals],
                                           feed_dict={PERTURBED_SIGNALS: per_sig})

    max_x = max(abs(transmitted_signal[0, :]))
    max_y = max(abs(transmitted_signal[1, :]))
    max_axis = 1.2 * max(max_x, max_y)

    pl.figure(figsize=(8, 8))
    pl.xlim(-max_axis, max_axis)
    pl.ylim(-max_axis, max_axis)
    pl.axis('equal')
    pl.scatter(transmitted_signal[0], transmitted_signal[1])
    pl.xlabel('X')
    pl.ylabel('Y')
    pl.grid()
    pl.savefig('transmitted_signals', bbox_inches='tight')

    color_map = cm.rainbow(np.linspace(0.0, 1.0, M))
    pl.figure(figsize=(8, 8))
    pl.grid()
    pl.axis('equal')
    pl.xlim(-max_axis, max_axis)
    pl.ylim(-max_axis, max_axis)
    for i in range(0, M):
        pl.scatter(power_con_sig[0, np.arange(i, num * M, M)], power_con_sig[1, np.arange(i, num * M, M)],
                   c=color_map[i], s=2)
    pl.xlabel('X')
    pl.ylabel('Y')
    pl.savefig('perturbed_signal')

    pl.figure(figsize=(8, 8))
    pl.grid()
    pl.axis('equal')
    for i in range(0, M):
        pl.scatter(fiber_signal[0, np.arange(i, num * M, M)], fiber_signal[1, np.arange(i, num * M, M)],
                   c=color_map[i], s=2)
    pl.xlabel('X')
    pl.ylabel('Y')
    pl.xlim(-max_axis, max_axis)
    pl.ylim(-max_axis, max_axis)
    pl.savefig('perturbed_fiber_signal')

    pl.figure(figsize=(8, 8))
    pl.pcolormesh(xx, yy, z)
    pl.xlim(-max_axis, max_axis)
    pl.ylim(-max_axis, max_axis)
    pl.axis('off')
    pl.savefig('Fiber_Optical_Decision_Region', bbox_inches='tight')






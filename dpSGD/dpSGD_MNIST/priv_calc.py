import tensorflow as tf
import accountant, utils
import argparse
import time
import sys

import pickle as pkl


def calc_priv(noise, epochs, training_size, batch_size):
    privacy_history = []
    with tf.Session() as sess:
        eps = tf.placeholder(tf.float32)
        delta = tf.placeholder(tf.float32)

        num_batches = (epochs+1) * (training_size / batch_size)
        target_eps = [0.125, 0.25, 0.5, 1, 2, 4, 8]
        priv_accountant = accountant.GaussianMomentsAccountant(training_size)

        sys.stderr.write('accum privacy, batches: ' + str(num_batches) + '\n')
        priv_start_time = time.clock()
        privacy_accum_op = priv_accountant.accumulate_privacy_spending(
          [None, None], args.noise, batch_size)
        tf.global_variables_initializer().run()

        for index in range(0, num_batches+1):
            sess.run([privacy_accum_op])
            with tf.control_dependencies([privacy_accum_op]):
                spent_eps_deltas = priv_accountant.get_privacy_spent(sess, target_eps=target_eps)

            if index % 6000 == 0:
                print(index, spent_eps_deltas)
                privacy_history.append(spent_eps_deltas)

        sys.stderr.write('priv time: ' + str(time.clock() - priv_start_time) +
                         '\n')

    pkl.dump(privacy_history, open('./privacy/' + str(noise) + '_' +
             str(epochs) + '_' + str(training_size) + '_' + str(batch_size) +
             '.p', 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--training_size", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    calc_priv(args.noise, args.epochs, args.training_size, args.batch_size)

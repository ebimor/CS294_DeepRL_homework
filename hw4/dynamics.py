import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """

        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
        self.st_at = tf.placeholder(shape=[None, ob_dim+ac_dim], name="input", dtype=tf.float32)
        self.delta = tf.placeholder(shape=[None, ob_dim], name="state", dtype=tf.float32)
        self.f_theta = build_mlp(self.st_at, ob_dim, "dynamics", n_layers, size, activation, output_activation)

        self.mu_s, self.sigma_s, self.mu_a, self.sigma_a, self.mu_delta, self.sigma_delta = normalization
        self.sess = sess
        self.batch_size = batch_size
        self.iter = iterations
        self.loss = tf.reduce_mean(tf.square(self.f_theta - self.delta))
        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-2).minimize(self.loss)


    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model
        going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
        states = np.concatenate([d['observations'] for d in data])
        actions = np.concatenate([d['actions'] for d in data])
        next_states = np.concatenate(d['next_observations'] for d in data)

        epsilon = 1e-6

        normalized_states = (states - self.mu_s)/(self.sigma_s+epsilon)
        normalized_next_states =  (next_states - self.mu_s)/(self.sigma_s+epsilon)
        normalized_actions =  (actions - self.mu_a)/(self.sigma_a+epsilon)

        normalized_deltas = normalized_next_states - normalized_states

        st_at_normalized = np.concatenate((normalized_states, normalized_actions), axis=1)

        for _ in range(self.iter):
            np.random.shuffle(indices)
            batches = int (math.ceil(normalized_state.shape[0] / self.batch_size))
            for i in range(batches):
                start_idx = i * self.batch_size
                idxs = indices[start_idx : start_idx + self.batch_size]
                batch_st_at = st_at_normalized[idxs, :]
                batch_delta = normalized_deltas[idxs, :]
                self.sess.run(self.trainer, feed_dict={self.st_at : batch_st_at, self.delta: batch_delta})


    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        epsilon = 1e-6

        normalized_states = (states - self.mu_s)/(self.sigma_s+epsilon)
        normalized_actions =  (actions - self.mu_a)/(self.sigma_a+epsilon)

        st_at_normalized = np.concatenate((normalized_states, normalized_actions), axis=1)

        normalized_delta = self.sess.run(self.f_theta, feed_dict={self.st_at : st_at_normalized})

        delta = self.mu_delta + normalized_delta*self.sigma_delta

        return delta + states

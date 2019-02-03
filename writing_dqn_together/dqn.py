import tensorflow as tf
import numpy as np
import gym

class ReplayBuffer:

    def __init__(self, obs_dim, size):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.actions_buf = np.zeros(size, dtype=np.int32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.actions_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    actions=self.actions_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

def mlp(x, hidden_sizes, activation=tf.tanh):
    for size in hidden_sizes:
        x = tf.layers.dense(x, units=size, activation=activation)
    return x

# (s,a,r,s')
# Q(s,a) <- r + gamma * max Q_targ(s',a')

def train(
        env_name='CartPole-v0', hidden_size=32, n_layers=1, gamma=0.99, lr=1e-3, total_steps=100_000,
        final_epsilon=0.05, replay_size=10_000, start_training_time=5_000, batch_size=32, steps_per_epoch=3000,
        n_test_episodes=10, n_test_render=2, target_update_interval=1_000):
    env = gym.make(env_name)
    test_env = gym.make(env_name)

    (obs_dim,) = env.observation_space.shape
    n_actions = env.action_space.n

    with tf.variable_scope('main') as scope:
        obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, obs_dim])
        final_hidden = mlp(obs_ph, hidden_sizes=[hidden_size]*n_layers)
        q_vals = tf.layers.dense(final_hidden, units=n_actions, activation=None)

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

    with tf.variable_scope('target') as scope:
        obs_targ_ph = tf.placeholder(dtype=tf.float32, shape=[None, obs_dim])
        final_hidden = mlp(obs_targ_ph, hidden_sizes=[hidden_size]*n_layers)
        q_vals_targ = tf.layers.dense(final_hidden, units=n_actions, activation=None)

        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

    rewards_ph = tf.placeholder(dtype=tf.float32, shape=[None])
    dones_ph = tf.placeholder(dtype=tf.bool, shape=[None])
    optimal_future_q = gamma * tf.reduce_max(q_vals_targ, axis=1)
    target = tf.stop_gradient(rewards_ph + (1 - tf.cast(dones_ph, dtype=tf.float32)) * optimal_future_q)
    actions_ph = tf.placeholder(dtype=tf.int32, shape=[None])
    predicted = tf.reduce_sum(q_vals * tf.one_hot(actions_ph, n_actions), axis=1)  # q_vals[actions_ph], except batched
    loss = tf.reduce_mean((target - predicted)**2, axis=0)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    update_target_op = tf.group(*[
        target_var.assign(main_var) for (target_var, main_var) in zip(target_vars, main_vars)])

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    def get_action(obs, eps):
        if np.random.random() < eps:
            return np.random.randint(0, n_actions), 0
        else:
            qs = sess.run(q_vals, {obs_ph: [obs]})
            action = np.argmax(qs, axis=1)[0]
            return action, qs[0,action]

    buffer = ReplayBuffer(obs_dim=obs_dim, size=replay_size)

    obs = env.reset()
    for t in range(total_steps):
        eps = (1 - t/total_steps) * (1.0 - final_epsilon) + final_epsilon
        action, _ = get_action(obs, eps)
        next_obs, reward, done, info = env.step(action)
        buffer.store(obs, action, reward, next_obs, done)
        if done:
            obs = env.reset()
        else:
            obs = next_obs

        if t > start_training_time:
            batch = buffer.sample_batch(batch_size)
            sess.run(train_op, {
                obs_ph: batch['obs'],
                actions_ph: batch['actions'],
                rewards_ph: batch['rews'],
                obs_targ_ph: batch['next_obs'],
                dones_ph: batch['done'],
            })

            if t % target_update_interval == 0:
                sess.run(update_target_op)

            if t % steps_per_epoch == 0:
                ep_lens = []
                q_list = []
                for ep in range(n_test_episodes):
                    test_obs = test_env.reset()
                    ep_len = 0
                    while True:
                        action, q = get_action(test_obs, final_epsilon)
                        q_list.append(q)
                        test_obs, test_rew, test_done, test_info = test_env.step(action)
                        ep_len += 1
                        if ep < n_test_render:
                            test_env.render()
                        if test_done:
                            break
                    ep_lens.append(ep_len)
                print(f"avg episode length: {np.mean(ep_lens)} \t avg q value: {np.mean(q_list)}")


if __name__ == '__main__':
    train()

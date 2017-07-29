from utils import *
import pickle
from osim.http.client import Client

class Actor:
    def __init__(self):
        self.env = LTR()
        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = np.prod(self.env.action_space.shape)
        self.hidden_size = 64
        weight_init = tf.random_uniform_initializer(-0.05, 0.05)
        bias_init = tf.constant_initializer(0)
        # tensorflow model of the policy
        self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
        with tf.variable_scope("policy-a"):
            h1 = fully_connected(self.obs, self.observation_size, self.hidden_size, weight_init, bias_init, "policy_h1")
            h1 = tf.nn.relu(h1)
            h2 = fully_connected(h1, self.hidden_size, self.hidden_size, weight_init, bias_init, "policy_h2")
            h2 = tf.nn.relu(h2)
            h3 = fully_connected(h2, self.hidden_size, self.action_size, weight_init, bias_init, "policy_h3")
            action_dist_logstd_param = tf.Variable((.01*np.random.randn(1, self.action_size)).astype(np.float32), name="policy_logstd")
        self.action_dist_mu = h3
        self.action_dist_logstd = tf.tile(action_dist_logstd_param, tf.stack((tf.shape(self.action_dist_mu)[0], 1)))

        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        self.session = tf.Session(config=config)
        self.session.run(tf.initialize_all_variables())
        var_list = tf.trainable_variables()

        self.set_policy = SetPolicyWeights(self.session, var_list)

    def act(self, obs):
        obs = np.expand_dims(obs, 0)
        action_dist_mu, action_dist_logstd = self.session.run([self.action_dist_mu, self.action_dist_logstd], feed_dict={self.obs: obs})
        # samples the guassian distribution
        act = action_dist_mu + np.exp(action_dist_logstd)*np.random.randn(*action_dist_logstd.shape)
        # act = action_dist_mu
        return act.ravel()


def test():
    with open('LTR-params.bin', 'rb') as f:
        policy = pickle.load(f)
    with open('filter-params-111.bin', 'rb') as f:
        filter_param = pickle.load(f)
    actor = Actor()
    actor.set_policy(policy)
    filter.load_state_dict(filter_param)

    env = LTR(visualize=False, difficulty=2)
    state = filter(env.reset())

    # env = RunEnv(visualize=False)
    # state = env.reset(difficulty=2, seed=np.random.randint(0, 10000000))
    done = False
    total_reward = 0.0
    while not done:
        action = actor.act(np.asarray(state))
        action = np.clip(action, 0, 1)
        state, reward, done, info = env.step(action)
        total_reward += reward
        state = filter(state)
    print total_reward

def submit():
    remote_base = "http://grader.crowdai.org:1729"
    crowdai_token = "0f526b098e86a8f6d91d3bc2af31b71b"
    client = Client(remote_base)

    with open('LTR-params.bin', 'rb') as f:
        policy = pickle.load(f)
    with open('filter-params-111.bin', 'rb') as f:
        filter_param = pickle.load(f)
    actor = Actor()
    actor.set_policy(policy)
    filter.load_state_dict(filter_param)

    def normalize_state(state):
        return filter(np.asarray(state) / math.pi)

    # Create environment
    state = client.env_create(crowdai_token)

    total_reward = 0.0
    while True:
        action = actor.act(normalize_state(state))
        [state, reward, done, info] = client.env_step(action.tolist(), True)
        if done:
            state = client.env_reset()
            if not state:
                break
    client.submit()

if __name__ == '__main__':
    # test()
    submit()


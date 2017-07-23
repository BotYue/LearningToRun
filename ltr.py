from async_agent import *
from DQN_agent import *
from DDPG_agent import *
import logging
import traceback
from random_process import *
import opensim as osim
from osim.env import RunEnv
from osim.http.client import Client
from osim.env import RunEnv
from logger import Logger
import math

class LTR(BasicTask):
    name = 'LearningToRun'
    success_threshold = 2000
    def __init__(self):
        BasicTask.__init__(self)
        self.env = RunEnv(visualize=False)

    def step(self, action):
        action = np.clip(action, 0, 1)
        next_state, reward, done, info = self.env.step(action)
        return np.asarray(next_state) / math.pi, reward, done, info

    def reset(self):
        state = self.env.reset(difficulty=0, seed=np.random.randint(0, 10000000))
        return np.asarray(state) / math.pi

def ddpg_agent():
    task_fn = lambda: LTR()
    task = task_fn()
    state_dim = task.env.observation_space.shape[0]
    action_dim = task.env.action_space.shape[0]
    config = dict()
    config['task_fn'] = task_fn
    config['actor_network_fn'] = lambda: DDPGActorNet(state_dim, action_dim, F.sigmoid)
    config['critic_network_fn'] = lambda: DDPGCriticNet(state_dim, action_dim)
    config['actor_optimizer_fn'] = lambda params: torch.optim.Adam(params, lr=1e-4)
    config['critic_optimizer_fn'] =\
        lambda params: torch.optim.Adam(params, lr=1e-3, weight_decay=0.01)
    config['replay_fn'] = lambda: HighDimActionReplay(memory_size=1000000, batch_size=64)
    config['discount'] = 0.99
    config['step_limit'] = 0
    config['tau'] = 0.001
    config['exploration_steps'] = 1000
    config['random_process_fn'] = \
        lambda: OrnsteinUhlenbeckProcess(size=action_dim, theta=0.15, sigma=0.2)
    config['test_interval'] = 100
    config['test_repetitions'] = 10
    config['tag'] = ''
    config['logger'] = Logger('./log', gym.logger, True)
    agent = DDPGAgent(**config)
    agent.run()

def test():
    task_fn = lambda: LTR()
    task = task_fn()
    state_dim = task.env.observation_space.shape[0]
    action_dim = task.env.action_space.shape[0]
    with open('data/ddpg-model-LearningToRun.bin', 'rb') as f:
        model = pickle.load(f)
    actor = DDPGActorNet(state_dim, action_dim)
    actor.load_state_dict(model)

    logger = Logger('./log')

    env = RunEnv(visualize=False)
    state = env.reset(difficulty=0)
    print state
    done = False
    total_reward = 0.0
    step = 0
    while not done:
        action = actor.predict(np.stack([state]), to_numpy=True).flatten()
        state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        logger.histo_summary('input', actor.input, step)
        logger.histo_summary('act1', actor.act1, step)
        logger.histo_summary('act2', actor.act2, step)
        logger.histo_summary('pre_act3', actor.pre_act3, step)
        logger.histo_summary('act3', actor.act3, step)
        for tag, value in actor.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.numpy(), step)

    print total_reward
    print step

def submit():
    remote_base = "http://grader.crowdai.org:1729"
    crowdai_token = "[YOUR_CROWD_AI_TOKEN_HERE]"
    client = Client(remote_base)

    task_fn = lambda: LTR()
    task = task_fn()
    state_dim = task.env.observation_space.shape[0]
    action_dim = task.env.action_space.shape[0]
    with open('data/ddpg-model-LearningToRun.bin', 'rb') as f:
        model = pickle.load(f)
    actor = DDPGActorNet(state_dim, action_dim)
    actor.load_state_dict(model)

    # Create environment
    state = client.env_create(crowdai_token)

    total_reward = 0.0
    while True:
        action = actor.predict(np.stack([state]), to_numpy=True).flatten()
        [state, reward, done, info] = client.env_step(action, True)
        total_reward += reward
        print(observation)
        if done:
            observation = client.env_reset()
            if not observation:
                break
    print total_reward
    client.submit()


if __name__ == '__main__':
    ddpg_agent()
    # test()
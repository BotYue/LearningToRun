from async_agent import *
from dqn_agent import *
from DDPG_agent import *
import logging
import traceback
from random_process import *
import opensim as osim
from osim.env import RunEnv

class LTR(BasicTask):
    success_threshold = 2000
    def __init__(self):
        BasicTask.__init__(self)
        self.env = RunEnv(visualize=False)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return np.asarray(next_state), reward, done, info

    def reset(self):
        state = self.env.reset()
        return np.asarray(state)

def ddpg_agent():
    task_fn = lambda: LTR()
    task = task_fn()
    state_dim = task.env.observation_space.shape[0]
    action_dim = task.env.action_space.shape[0]
    config = dict()
    config['task_fn'] = task_fn
    config['actor_network_fn'] = lambda: DDPGActorNet(state_dim, action_dim)
    config['critic_network_fn'] = lambda: DDPGCriticNet(state_dim, action_dim)
    config['actor_optimizer_fn'] = lambda params: torch.optim.Adam(params, lr=1e-4)
    config['critic_optimizer_fn'] =\
        lambda params: torch.optim.Adam(params, lr=1e-3, weight_decay=0.01)
    config['replay_fn'] = lambda: HighDimActionReplay(memory_size=1000000, batch_size=64)
    config['discount'] = 0.99
    config['step_limit'] = 200
    config['tau'] = 0.001
    config['exploration_steps'] = 100
    config['random_process_fn'] = \
        lambda: OrnsteinUhlenbeckProcess(size=action_dim, theta=0.15, sigma=0.2)
    config['test_interval'] = 10
    config['test_repetitions'] = 10
    config['tag'] = ''
    config['logger'] = gym.logger
    agent = DDPGAgent(**config)
    agent.run()

if __name__ == '__main__':
    ddpg_agent()

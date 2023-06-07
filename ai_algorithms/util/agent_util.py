from ..a3c import A3CAgent
from ..dqn import DQNAgent
from ..ppo import PPOAgent

def create_agent(agent, env):
    agents = {
        'a3c': A3CAgent(env),
        'dqn': DQNAgent(env),
        'ppo': PPOAgent(env),
    }

    return agents[agent]
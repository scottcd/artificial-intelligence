from ..a3c import A3CAgent
from ..dqn import DQNAgent
from ..ppo import PPOAgent

def create_agent(agent):
    agents = {
        'a3c': A3CAgent(),
        'dqn': DQNAgent(),
        'ppo': PPOAgent(),
    }

    return agents[agent]
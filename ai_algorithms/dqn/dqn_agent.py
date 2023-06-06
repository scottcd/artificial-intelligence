from ..util.agent import Agent

class DQNAgent(Agent):
    def __init__(self) -> None:
        super().__init__()  
        self.name = 'dqn'
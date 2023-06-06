from ..util.agent import Agent

class PPOAgent(Agent):
    def __init__(self) -> None:
        super().__init__()  
        self.name = 'ppo'
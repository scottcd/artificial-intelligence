from ..util.agent import Agent

class PPOAgent(Agent):
    def __init__(self, env) -> None:
        super().__init__(env)  
        self.name = 'ppo'
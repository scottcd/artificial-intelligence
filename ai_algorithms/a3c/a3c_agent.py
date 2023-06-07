from ..util.agent import Agent

class A3CAgent(Agent):
    def __init__(self, env) -> None:
        super().__init__(env)      
        self.name = 'a3c'
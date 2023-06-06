from ..util.agent import Agent

class A3CAgent(Agent):
    def __init__(self) -> None:
        super().__init__()      
        self.name = 'a3c'
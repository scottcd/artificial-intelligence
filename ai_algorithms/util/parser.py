import argparse

class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='My Awesome Command-line Tool')
        self.parser.add_argument('-e', '--environment', help='Environment to train on', default='LunarLander-v2')
        self.parser.add_argument('-a', '--ml_algorithm', help='Machine learning algorithm', default='ppo')
        self.parser.add_argument('-n', '--n_episodes', help='Number of episodes', default=1000)

    def parse_args(self):
        return self.parser.parse_args()
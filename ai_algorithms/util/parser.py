import argparse

class ArgParser:
    def __init__(self):
        self.envs = {
            'lunar': 'LunarLander-v2',
            'walker': 'BipedalWalker-v3',
            'racing': 'CarRacing-v2',
            'blackjack': 'Blackjack-v1',
            'cliff_walking': 'CliffWalking-v0',
            'frozen_lake': 'FrozenLake-v1',
            'taxi': 'Taxi-v3',
            'acrobot': 'Acrobot-v1',
            'cart_pole': 'CartPole-v1',
            'mountain_car_cont': 'MountainCarContinuous-v0',
            'mountain_car': 'MountainCar-v0',
            'pendulum': 'Pendulum-v1',
        }
        self.parser = argparse.ArgumentParser(description='Library to train PPO, DQN, and A3C on multiple gymnasium environments.')
        self.parser.add_argument('-e', '--environment', help='Environment to train ons', 
                                 choices=self.envs.keys(), default='walker')
        self.parser.add_argument('-a', '--ml_algorithm', help='Machine learning algorithm to train on the environment.', 
                                 choices=['ppo','dqn','a3c'], default='ppo')
        self.parser.add_argument('-n', '--n_episodes', help='Number of episodes to train', default=1000)
        self.parser.add_argument('-l', '--logs', help='File name for log output', default=None)
        self.parser.add_argument('-s', '--stats', help='File name for statistics output', default=None)

        self.args = self.parser.parse_args()
    
    def get_algorithm(self):
        return self.args.ml_algorithm

    def get_environment(self):
        env = self.args.environment


        if env in self.envs:
            return self.envs[env]
        else:
            raise Exception('Environment not found in the dictionary')

    
    def get_n_episodes(self):
        return self.args.n_episodes
    
    def get_logs_and_stats(self):
        return (self.args.logs, self.args.stats)
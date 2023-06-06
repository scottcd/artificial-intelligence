import sys
import gymnasium as gym
from .util import CustomLogger as log, ArgParser as parser

if __name__ == '__main__':
    # initialize logger
    logger = log()

    # get args
    logger.info('Parsing args..')
    arg_parser = parser()
    args = arg_parser.parse_args()
    logger.info(f'Args parsed: ({args})')

    ml_algorithm = args.ml_algorithm
    environment = args.environment
    n_episodes = args.n_episodes

    # set up training
    logger.info('Setting up environment..')
    env = gym.make(environment, render_mode="human")
    observation, info = env.reset(seed=42)
    logger.info('Environment set up.')

    # train
    logger.info(f'Training a {ml_algorithm} agent for {n_episodes} episodes in the {environment} environment.')
    ##
    for i in range(n_episodes):
        logger.info(f'Training on episode {i}')
        while True:    
            action = env.action_space.sample()  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()
                break
    env.close()
    ##
    logger.info(f'Finished training!')


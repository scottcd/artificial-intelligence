import sys
import gymnasium as gym
from .util import CustomLogger as log, ArgParser, create_agent



if __name__ == '__main__':
    # get args
    arg_parser = ArgParser()
    ml_algorithm = arg_parser.get_algorithm()
    environment = arg_parser.get_environment()
    n_episodes = arg_parser.get_n_episodes()
    logs, stats = arg_parser.get_logs_and_stats()

    # initialize logger
    logger = log(log_file=logs, stats_file=stats)
    logger.info(f'{ml_algorithm} {environment} {n_episodes}')

    # set up training
    logger.info('Setting up environment..')
    env = gym.make(environment, render_mode="human")
    observation, info = env.reset(seed=42)
    logger.info('Environment set up.')

    logger.info(f'Setting up {ml_algorithm.upper()} agent..')
    agent = create_agent(ml_algorithm)
    logger.critical(f'{agent.memory}')
    logger.info('Agent set up.')

    # train
    logger.info(f'Training a(n) {ml_algorithm.upper()} agent for {n_episodes} episodes in the {environment} environment.')
    ##
    for i in range(n_episodes):
        logger.info(f'Training on episode {i}')
        while True:    
            action = env.action_space.sample()  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                logger.info(f'Episode {i} complete with reward {reward}.')
                observation, info = env.reset()
                break
    env.close()
    ##
    logger.info(f'Finished training!')


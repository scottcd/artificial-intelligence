import sys
import torch
import gymnasium as gym
from .util import CustomLogger as log, ArgParser, create_agent
from .a3c import A3CAgent
from .dqn import DQNAgent
from .ppo import PPOAgent


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
    logger.info((
        f'Environment: {env.unwrapped.spec.id}\n'
        f'Observation Space: {"discrete" if isinstance(env.observation_space, gym.spaces.Discrete) else "continuous"} '
        f'{env.observation_space}\n'
        f'Action Space: {"discrete" if isinstance(env.action_space, gym.spaces.Discrete) else "continuous"} '
        f'{env.action_space}\n'
    ))

    logger.info(f'Setting up {ml_algorithm.upper()} agent..')
    agent = create_agent(ml_algorithm, env)
    logger.info('Agent set up.')

    # train
    logger.info(
        f'Training a(n) {ml_algorithm.upper()} agent for {n_episodes} episodes in the {environment} environment.')
    ##
    for i in range(n_episodes):
        logger.info(f'Training on episode {i}')
        
        observation, info = env.reset()
        state = torch.tensor(observation, dtype=torch.float32, device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")).unsqueeze(0)
        total_reward = 0
        while True:
            model = agent.model if type(agent) is A3CAgent or type(agent) is PPOAgent \
                else agent.policy_network

            action = agent.select_action(model, state)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")).unsqueeze(0)

            # if memory, remember
            if agent.memory is not None:
                agent.remember(state, next_state, torch.tensor(reward).unsqueeze(
                    0), torch.tensor(action).unsqueeze(0).unsqueeze(0))
                state = next_state

            # optimize
            agent.learn()

            total_reward += reward
            if done:
                # stats for episode
                logger.info(
                    f'Episode {i} complete with reward {total_reward}.')
                break
    env.close()
    ##
    logger.info(f'Finished training!')

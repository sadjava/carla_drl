import os
import sys
import time
import random
import numpy as np
import argparse
import logging
import pickle
import torch
from distutils.util import strtobool
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import carla

from carla_drl.lane_following.encoder import ObservationEncoder
from carla_drl.lane_following.agent import PPOAgent
from carla_drl.lane_following.parameters import *   
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='im20_480x270', help='name of the experiment')
    parser.add_argument('--ss_model_path', type=str, default=SS_MODEL_PATH, help='path to semantic segmentation model')
    parser.add_argument('--depth_model_path', type=str, default=DEPTH_MODEL_PATH, help='path to depth estimation model')
    parser.add_argument('--image_size', type=int, nargs=2, default=[480, 270], help='image size')
    
    parser.add_argument('--train', default=True, type=bool, help='is it training?')
    parser.add_argument('--learning-frequency', type=int, default=LEARN_FREQUENCY, help='learning frequency')
    parser.add_argument('--learning-rate', type=float, default=PPO_LEARNING_RATE, help='learning rate of the optimizer')
    parser.add_argument('--total-timesteps', type=int, default=TOTAL_TIMESTEPS, help='total timesteps of the experiment')
    parser.add_argument('--test-timesteps', type=int, default=TEST_TIMESTEPS, help='timesteps to test our model')
    parser.add_argument('--episode-length', type=int, default=EPISODE_LENGTH, help='max timesteps in an episode')

    parser.add_argument('--env-checkpoint-frequency', type=int, default=ENV_CHECKPOINT_FREQUENCY, help='frequency of checkpointing route')
    parser.add_argument('--logging-frequency', type=int, default=LOGGING_FREQUENCY, help='frequency of logging')
    parser.add_argument('--map-frequency', type=int, default=MAP_FREQUENCY, help='frequency of changing map')
    parser.add_argument('--weather-frequency', type=int, default=WEATHER_FREQUENCY, help='frequency of changing weather')

    parser.add_argument('--checkpoint-dir', type=str, default=PPO_CHECKPOINT_DIR, help='path for saving checkpoint folder')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='path to checkpoint file if resume training')

    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--seed', type=int, default=SEED, help='seed of the experiment')
    args = parser.parse_args()
    
    return args


def runner():

    #========================================================================
    #                           BASIC PARAMETER & LOGGING SETUP
    #========================================================================
    
    args = parse_args()
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    if not args.checkpoint_path:
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        log_dir = os.path.join(args.checkpoint_dir, args.exp_name, timestamp)
    else:
        log_dir = args.checkpoint_path

    writer = SummaryWriter(os.path.join(log_dir))

    os.makedirs(os.path.join(log_dir, 'policy'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'checkpoint'), exist_ok=True)

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])))


    #Seeding to reproduce the results 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    
    timestep = 0
    episode = 0
    cumulative_score = 0
    episodic_length = list()
    scores = list()
    deviation_from_center = 0
    distance_covered = 0

    #========================================================================
    #                           CREATING THE SIMULATION
    #========================================================================
    weather_dict = dict([(name, param) for name, param in carla.WeatherParameters.__dict__.items() if isinstance(param, carla.WeatherParameters)])
    town = random.choice(TOWN_LIST)
    try:
        client, world = ClientConnection(town).setup()
        logging.info("Connection has been setup successfully.")
    except:
        logging.error("Connection has been refused by the server.")
        ConnectionRefusedError

    if not args.train:
        args.env_checkpoint_frequency = None
    env = CarlaEnvironment(client, world, town, checkpoint_frequency=args.env_checkpoint_frequency)
    encoder = ObservationEncoder(args.ss_model_path, args.depth_model_path, (args.image_size[1], args.image_size[0]))


    #========================================================================
    #                           ALGORITHM
    #========================================================================
    try:
        time.sleep(0.5)
        agent = PPOAgent(
            ss_channels=encoder.ss_model.enc4.out_channels,
            depth_channels=encoder.depth_model.scratch.layer4_rn.out_channels,
            navigation_dim=5, device=device
        )

        if args.checkpoint_path:
            agent.load(os.path.join(args.checkpoint_path, 'policy', AGENT_FILE))
        
            with open(os.path.join(args.checkpoint_path, 'checkpoint', CHECKPOINT_FILE), 'rb') as f:
                data = pickle.load(f)
                episode = data['episode']
                timestep = data['timestep']
                cumulative_score = data['cumulative_score']

        if not args.train:
            agent.load(os.path.join(args.checkpoint_path, 'policy', AGENT_FILE))
            for params in agent.old_policy.actor.parameters():
                params.requires_grad = False

        if args.train:
            #Training
            while timestep < args.total_timesteps:
            
                observation = env.reset()
                observation = encoder.process(observation)

                current_ep_reward = 0
                t1 = datetime.now()

                for t in range(args.episode_length):
                
                    # select action with policy
                    action = agent.get_action(observation, train=True)

                    observation, reward, done, info = env.step(action)
                    if observation is None:
                        break
                    observation = encoder.process(observation)
                    
                    agent.memory.rewards.append(reward)
                    agent.memory.dones.append(done)
                    
                    timestep +=1
                    current_ep_reward += reward

                    # break; if the episode is over
                    if done:
                        episode += 1

                        t2 = datetime.now()
                        t3 = t2 - t1
                        
                        episodic_length.append(abs(t3.total_seconds()))
                        break
                
                deviation_from_center += info[1]
                distance_covered += info[0]
                
                scores.append(current_ep_reward)
                
                if args.checkpoint_path:
                    cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / (episode)
                else:
                    cumulative_score = np.mean(scores)


                print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score))
                if episode % args.learning_frequency == 0:
                    agent.learn()
                    agent.save(os.path.join(log_dir, 'policy', AGENT_FILE))
                    chkpt_file = os.path.join(log_dir, 'checkpoint', CHECKPOINT_FILE)
                    data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)
                    
                
                if episode % args.logging_frequency == 0:

                    writer.add_scalar("Episodic Reward/episode", scores[-1], episode)
                    writer.add_scalar("Cumulative Reward/info", cumulative_score, episode)
                    writer.add_scalar("Cumulative Reward/(t)", cumulative_score, timestep)
                    writer.add_scalar("Average Episodic Reward/info", np.mean(scores[-5]), episode)
                    writer.add_scalar("Average Reward/(t)", np.mean(scores[-5]), timestep)
                    writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), episode)
                    writer.add_scalar("Reward/(t)", current_ep_reward, timestep)
                    writer.add_scalar("Average Deviation from Center/episode", deviation_from_center / args.logging_frequency, episode)
                    writer.add_scalar("Average Deviation from Center/(t)", deviation_from_center / args.logging_frequency, timestep)
                    writer.add_scalar("Average Distance Covered (m)/episode", distance_covered / args.logging_frequency, episode)
                    writer.add_scalar("Average Distance Covered (m)/(t)", distance_covered / args.logging_frequency, timestep)

                    episodic_length = list()
                    deviation_from_center = 0
                    distance_covered = 0
                
                if episode % args.map_frequency == 0:
                    env.change_town(random.choice(TOWN_LIST))
                if episode % args.weather_frequency == 0:
                    env.change_weather(weather_dict[random.choice(WEATHER_LIST)])
                        
            print("Terminating the run.")
            sys.exit()
        else:
            #Testing
            while timestep < args.test_timesteps:
                observation = env.reset()
                observation = encoder.process(observation)

                current_ep_reward = 0
                t1 = datetime.now()
                for t in range(args.episode_length):
                    # select action with policy
                    action = agent.get_action(observation, train=False)
                    observation, reward, done, info = env.step(action)
                    if observation is None:
                        break
                    observation = encoder.process(observation)
                    
                    timestep +=1
                    current_ep_reward += reward
                    # break; if the episode is over
                    if done:
                        episode += 1

                        t2 = datetime.now()
                        t3 = t2 - t1
                        
                        episodic_length.append(abs(t3.total_seconds()))
                        break
                deviation_from_center += info[1]
                distance_covered += info[0]
                
                scores.append(current_ep_reward)
                cumulative_score = np.mean(scores)

                print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score))
                
                writer.add_scalar("TEST: Episodic Reward/episode", scores[-1], episode)
                writer.add_scalar("TEST: Cumulative Reward/info", cumulative_score, episode)
                writer.add_scalar("TEST: Cumulative Reward/(t)", cumulative_score, timestep)
                writer.add_scalar("TEST: Episode Length (s)/info", np.mean(episodic_length), episode)
                writer.add_scalar("TEST: Reward/(t)", current_ep_reward, timestep)
                writer.add_scalar("TEST: Deviation from Center/episode", deviation_from_center, episode)
                writer.add_scalar("TEST: Deviation from Center/(t)", deviation_from_center, timestep)
                writer.add_scalar("TEST: Distance Covered (m)/episode", distance_covered, episode)
                writer.add_scalar("TEST: Distance Covered (m)/(t)", distance_covered, timestep)

                episodic_length = list()
                deviation_from_center = 0
                distance_covered = 0

            print("Terminating the run.")
            sys.exit()

    finally:
        sys.exit()


if __name__ == "__main__":
    try:        
        runner()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nExit')
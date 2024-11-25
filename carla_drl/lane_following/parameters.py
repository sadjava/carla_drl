SEED = 42

SS_MODEL_PATH = 'results/semantic_segmentation/t2_wcn_im2000/epoch_497_loss_0.01070_acc_0.94854_acc-cls_0.75613_mean-iu_0.69873_fwavacc_0.90440_lr_0.0015009464.pth'
SS_AE_MODEL_PATH = 'results/autoencoder-semseg/ae_semseg.pth'
DE_AE_MODEL_PATH = 'results/autoencoder-depth/ae_depth.pth'

LATENT_DIMS = 50

AGENT_FILE = 'last_agent.pth'
CHECKPOINT_FILE = 'last_chekpoint.pickle'

ACTION_STD_INIT = 0.2
ACTION_STD_DECAY_RATE = 0.05
MIN_ACTION_STD = 0.05
ACTION_STD_DECAY_FREQ = 5e5

EPISODE_LENGTH = 7500
TOTAL_TIMESTEPS = 2e6
LEARN_FREQUENCY = 10 # episodes
TEST_TIMESTEPS = 5e4
PPO_LEARNING_RATE = 1e-4  
PPO_CHECKPOINT_DIR = 'results/lane_following/ppo/'
POLICY_EPSILON = 0.2
GAMMA = 0.99

ENV_CHECKPOINT_FREQUENCY = 100 # steps
LOGGING_FREQUENCY = 5 # episodes
MAP_FREQUENCY = 50 # episodes
WEATHER_FREQUENCY = 20 # episodes

TOWN_LIST = ['Town01_Opt', 'Town02_Opt', 'Town03_Opt', 'Town04_Opt', 'Town05_Opt', 'Town10HD_Opt']
WEATHER_LIST = ['Default', 'ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'MidRainyNoon', 
                'HardRainNoon', 'SoftRainNoon', 'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset', 
                'MidRainSunset', 'HardRainSunset', 'SoftRainSunset', 'ClearNight', 'CloudyNight', 'WetNight', 'WetCloudyNight', 
                'SoftRainNight', 'MidRainyNight', 'HardRainNight', 'DustStorm']
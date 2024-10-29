SEED = 42
SS_MODEL_PATH = 'results/semantic_segmentation/im20_480x270/epoch_215_loss_0.01857_acc_0.91265_acc-cls_0.69529_mean-iu_0.60999_fwavacc_0.84527_lr_0.0100000000.pth'
DEPTH_MODEL_PATH = 'results/depth_estimation/im20_480x270/epoch_97_loss_0.00136_mae_0.05542_rmse_0.14633_relative-error_1.89326_lr_0.0010000000.pth'

AGENT_FILE = 'last_agent.pth'
CHECKPOINT_FILE = 'last_chekpoint.pickle'

EPISODE_LENGTH = 7500
TOTAL_TIMESTEPS = 2e6
LEARN_FREQUENCY = 10 # episodes
TEST_TIMESTEPS = 5e4
PPO_LEARNING_RATE = 1e-4  
PPO_CHECKPOINT_DIR = 'results/lane_following/ppo/'
POLICY_EPSILON = 0.2

ENV_CHECKPOINT_FREQUENCY = 100 # steps
LOGGING_FREQUENCY = 5 # episodes
MAP_FREQUENCY = 50 # episodes
WEATHER_FREQUENCY = 1 # episodes

TOWN_LIST = ['Town01_Opt', 'Town02_Opt', 'Town03_Opt', 'Town04_Opt', 'Town05_Opt', 'Town10HD_Opt']
WEATHER_LIST = ['Default', 'ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'MidRainyNoon', 
                'HardRainNoon', 'SoftRainNoon', 'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset', 
                'MidRainSunset', 'HardRainSunset', 'SoftRainSunset', 'ClearNight', 'CloudyNight', 'WetNight', 'WetCloudyNight', 
                'SoftRainNight', 'MidRainyNight', 'HardRainNight', 'DustStorm']
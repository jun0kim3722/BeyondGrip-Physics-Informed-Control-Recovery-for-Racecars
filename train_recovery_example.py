# In order to use the recovery environmet import the following that is all that is necessary
from AssettoCorsaEnv.recovery_env import RecoveryAssettoEnv


# Other imports that you need such as the algoirhtms, etc.
from discor.algorithm import SAC, DisCor, DDPG
import AssettoCorsaEnv.assettoCorsa as assettoCorsa



# just to remove errors
config = 1
work_dir = 1


# This code says to make an environment, but use the RecoveryAssettoEnv instead of AssettoCorsaEnv
# An episode is terminated upon 2 conditions: previous termination conditions such as crashing
# OR a slip recovery which is maintaining a slip angle of less than 7 degrees for 1 second.
# This may be too long, but just put as an arbitrary value.

# To use your own rewards, it is simple, you can inherrit RecoveryAssetoEnv again and then just override the two functions
# for dense_reward and terminal_reward.

# See assetto_corsa_gym/AssettoCorsaEnv/recovery_ac_env.py for comments on how that works and more specifics


# The updates will not break any of the prewritten code (in theory).
# If you do not env_class, it will default to using the AssettoEnv (non-recovery) and its associated rewards.

# You can use this newly created environment as you would any other environment
env = assettoCorsa.make_ac_env(
        cfg=config,
        work_dir=work_dir,
        env_class=RecoveryAssettoEnv,        # <- IMPORTANT
        env_kwargs=dict(                     
            slip_threshold=7,  # slip threshold measured in angles
            recovery_time=1.0, # time in seconds
        )
    )


# when you do something like this it automatically passes you the reward that you setup in the RecoveryEnvironment
# based on whether it should be a terminal reward (crash vs not) or a per-step reward.
next_state, reward, done, info = env.step(action=None)  # action is already applied

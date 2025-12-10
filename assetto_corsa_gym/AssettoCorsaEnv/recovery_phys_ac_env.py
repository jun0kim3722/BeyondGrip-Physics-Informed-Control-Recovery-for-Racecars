from AssettoCorsaEnv.recovery_env import RecoveryAssettoEnv
import numpy as np

class PhysicsRecoveryEnv(RecoveryAssettoEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def dense_reward(self, state, actions_diff, info):
        """
        This just calls your exssiting physics reward function.
        You will probably want to move your logic out of ac_env and just put it here.
        See the base class recovery_ac_env.py for more ways you can use this.
        """
        r = self.get_physics_reward(state)

        return np.array([r]).reshape(-1)
    
    # Keeps the same terminal reward. See recovery_ac_env.py for range of terminal reward.
    # def terminal_reward(self, state, info):


from AssettoCorsaEnv.recovery_env import RecoveryAssettoEnv
from AssettoCorsaEnv.ac_env import logger

import numpy as np

class PhysicsRecoveryEnv(RecoveryAssettoEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_reward(self, state, actions_diff, info):
        r = super().get_reward(state, actions_diff, info)

        # call original reward, but also call your debugging.
        if (self.ep_steps % 50) == 0:
                    logger.debug(f't: {self.ep_steps} speed: {state["speed"]:.2f}, oot: {state["out_of_track"]} '
                                f's: {self.actions[0]:.2f} a: {self.actions[1]:.2f} b: {self.actions[2]:.2f} '
                                f'reward: {state["reward"]:.3f} '
                                f'done: {state["done"]:.0f} LapDist: {state["LapDist"]:.0f} gap: {state["gap"]:.1f} '
                                )
        return r


    
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


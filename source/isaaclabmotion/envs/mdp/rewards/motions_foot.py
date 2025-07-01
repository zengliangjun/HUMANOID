import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from isaaclabmotion.envs.managers import motions_manager
import isaaclab.utils.math as math_utils
from extends.isaac_utils import rotations

"""
isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py
feet_air_time
"""
class Base(ManagerTermBase):
    """Base class for joint statistics calculation with common functionality."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        motions_name: str = cfg.params["motions_name"]
        self.motions: motions_manager.MotionsTerm = env.motions_manager.get_term(motions_name)

    def _reward_motion(self) -> dict:
        return self.motions.motion_ref(0)

class reward_motion_feet_air_time(Base):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg,
        motions_name: str,
        threshold: float = 0.25,
    ) -> torch.Tensor:

        """Reward long steps taken by the feet using L2-kernel.

        This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
        that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
        the time for which the feet are in the air.

        If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
        """
        # extract the used quantities (to enable type-hinting)
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        # compute the reward
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
        # no reward for zero command
        ref_motions = self._reward_motion()
        motions_lin = ref_motions['root_vel'][:, : 2]

        reward *= torch.norm(motions_lin, dim=1) > 0.1
        return reward

from isaaclab.utils import configclass
import math
from dataclasses import MISSING

from isaaclab.managers import EventTermCfg, TerminationTermCfg, ObservationTermCfg, \
                              ObservationGroupCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclabex.envs.mdp.observations import privileged

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""
        ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        gravity = ObservationTermCfg(func=mdp.projected_gravity)
        commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        actions = ObservationTermCfg(func=mdp.last_action)

        ## proprioceptive
        lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        contact_forces = ObservationTermCfg(
            func=privileged.feet_contact_forces,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
            }
        )
        payload = ObservationTermCfg(
            func=privileged.body_mass,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        stiffness = ObservationTermCfg(
            func=privileged.joint_stiffness,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
        )

        damping = ObservationTermCfg(
            func=privileged.joint_damping,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
        )
        # discrete
        contact_status = ObservationTermCfg(
            func=privileged.feet_contact_status,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "threshold": 1.0,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


from __future__ import annotations

from isaaclab.utils import configclass
from isaaclabex.scenes import scenes_cfg
from isaaclab.envs import mdp

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclabmotion.envs import env_motions_cfg
from isaaclabmotion.assets.motions import asap_omnih2oh1


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1, use_default_offset=False)
    joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*"], scale=1, use_default_offset=False)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        # observation terms (order preserved)
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class MotionsCfg:

    omnih2o = asap_omnih2oh1.OMNIH2OH1_CFG


@configclass
class OMNIH2OH1EnvCfg(env_motions_cfg.ManagerMotionsCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = scenes_cfg.BaseSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    #
    actions = ActionsCfg()
    #
    motions = MotionsCfg()


    def __post_init__(self):
        """Post initialization."""
        self.scene.height_scanner = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # general settings
        self.decimation = 4        ##
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005        ## 1 / 200
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if hasattr(self.scene, "height_scanner") and self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
            self.scene.height_scanner.prim_path="{ENV_REGEX_NS}/Robot/base"
        if hasattr(self.scene, "contact_forces") and self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        self.motions.omnih2o.motion_file = f"{asap_omnih2oh1._ASSETS_ROOT}/motions/asap/omnih2o/amass_phc_filtered.pkl"
        self.motions.omnih2o.step_dt = self.decimation * self.sim.dt
        self.motions.omnih2o.random_sample = True
        self.motions.omnih2o.debug_vis = True

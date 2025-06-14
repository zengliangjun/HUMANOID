from isaaclabex.assets.robots import unitree_g112
from isaaclabex.scenes import scenes_cfg
from isaaclabex.envs import rl_env_exts_cfg
from isaaclabex.envs.mdp.commands import commands_cfg
from isaaclab.utils import configclass
from ..mdps  import mdps, curriculum, obs, events
from . import rewards

from isaaclabex.envs.mdp.curriculum import rewards as rewards_curriculum
from isaaclab.managers import CurriculumTermCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclabex.terrains.config import rough_low_level_cfg


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = commands_cfg.ZeroSmallCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=commands_cfg.ZeroSmallCommandCfg.Ranges(
            #lin_vel_x=(0, 4.5), lin_vel_y=(-0.75, 0.75), ang_vel_z=(-2., 2.), heading=(0., 0)
            lin_vel_x=(0, 2.8), lin_vel_y=(-0.35, 0.35), ang_vel_z=(-2., 2.), heading=(0., 0)
        ),
        small2zero_threshold_line=0.25,
        small2zero_threshold_angle=0.25
    )

    def __post_init__(self):
        self.base_velocity.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
        self.base_velocity.current_vel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)

@configclass
class ObservationsCfg(obs.ObservationsCfg):

    def __post_init__(self):
        self.policy.phase = None
        self.critic.phase = None

@configclass
class CurriculumCfg(curriculum.CurriculumCfg):

    terrain_levels = CurriculumTermCfg(func=mdp.terrain_levels_vel)

    penalize_with_steps = CurriculumTermCfg(
        func=rewards_curriculum.curriculum_with_steps,
        params={
            'start_steps': 0,
            'end_steps': 800000,
            "curriculums": {
                'p_action_smoothness': {    # reward name
                    "start_weight": -0.004,
                    "end_weight": -0.02
                },
                'p_torques': {    # reward name
                    "start_weight": -0.0005,
                    "end_weight": -0.001
                },
                'p_width': {    # reward name
                    "start_weight": -3.0,
                    "end_weight": -10
                },
                'p_orientation': {    # reward name
                    "start_weight": -1.0,
                    "end_weight": -10
                },
                'p_height': {    # reward name
                    "start_weight": -10.0,
                    "end_weight": -40.0
                }
            }
        }
    )


@configclass
class G1FlatEnvCfg(rl_env_exts_cfg.ManagerBasedRLExtendsCfg):
    # Scene settings
    scene = scenes_cfg.BaseSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions =  mdps.ActionsCfg()
    commands = CommandsCfg()
    # MDP settings
    rewards = rewards.RewardsCfg()
    terminations = mdps.TerminationsCfg()
    events = events.EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        # ROBOT
        self.scene.robot = unitree_g112.UNITREE_GO112_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        #self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = rough_low_level_cfg.ROUGH_TERRAINS_CFG
        self.scene.height_scanner = None

        self.events.base_external_force_torque = None

        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.curriculum = None

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.commands.base_velocity.resampling_time_range = (4.0, 4.0)
        self.episode_length_s = 12.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.size=(6.0, 6.0)
            self.scene.terrain.terrain_generator.num_rows = 6
            self.scene.terrain.terrain_generator.num_cols = 6
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing


from isaaclabex.envs.mdp.statistics import joints
from isaaclabex.envs.managers import term_cfg
from isaaclab.managers import SceneEntityCfg
from . import rewardsv2

@configclass
class StatisticsCfg:
    pos = term_cfg.StatisticsTermCfg(
        func= joints.StatusJPos,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot")},
        export_interval = 1000000
    )


@configclass
class G1FlatEnvV2Cfg(G1FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.statistics = StatisticsCfg()
        self.rewards = rewardsv2.RewardsCfg()


@configclass
class G1FlatEnvV2Cfg_PLAY(G1FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.statistics = StatisticsCfg()
        self.rewards = rewardsv2.RewardsCfg()

from . import rewardsv3


@configclass
class CurriculumCfgv3(CurriculumCfg):

    rew_with_steps = CurriculumTermCfg(
        func=rewards_curriculum.curriculum_with_steps,
        params={
            'start_steps': 0,
            'end_steps': 80000,
            "curriculums": {
                'rew_mean_hip': {    # reward name
                    "start_weight": 0.8,
                    "end_weight": 0.6
                },
                'rew_mean_knee': {    # reward name
                    "start_weight": 0.8,
                    "end_weight": 0.6
                },
                'rew_variance_self': {    # reward name
                    "start_weight": 0.8,
                    "end_weight": 0.6
                },
                'p_foot_clearance': {    # reward name
                    "start_weight": -20.0,
                    "end_weight": -80.0
                }
            }
        }
    )

@configclass
class G1FlatEnvV3Cfg(G1FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.curriculum = CurriculumCfgv3()
        self.statistics = StatisticsCfg()
        self.rewards = rewardsv3.RewardsCfg()

@configclass
class G1FlatEnvV3Cfg_PLAY(G1FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.statistics = StatisticsCfg()
        self.rewards = rewardsv3.RewardsCfg()

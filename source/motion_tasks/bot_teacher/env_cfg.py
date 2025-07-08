from isaaclab.utils import configclass
from isaaclabex.assets.robots import unitree_hoverh1
from isaaclabex.scenes import scenes_cfg

from .mdps  import mdps, obs, events, rewards, curriculum

from isaaclabmotion.envs import rl_env_motions_cfg
from isaaclabmotion.assets.motions import hover_h1

@configclass
class MotionsCfg:
    hoverh1 = hover_h1.HOVERH1_CFG


@configclass
class BOTH1Cfg(rl_env_motions_cfg.RLMotionsENVCfg):
    # Scene settings
    scene = scenes_cfg.BaseSceneCfg(num_envs=4096, env_spacing=4)
    # Basic settings
    observations: obs.ObservationsCfg = obs.ObservationsCfg()
    actions =  mdps.ActionsCfg()
    # MDP settings
    rewards = rewards.RewardsCfg()
    terminations = mdps.TerminationsCfg()
    events = events.EventCfg()
    curriculum = curriculum.CurriculumCfg()

    motions = MotionsCfg()

    statistics = mdps.StatisticsCfg()

    def __post_init__(self):
        self.scene.height_scanner = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

         # ROBOT
        self.scene.robot = unitree_hoverh1.H1HOVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


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

        # motion settings
        self.motions.hoverh1.motion_file = f"{hover_h1._ASSETS_ROOT}/motions/asap/omnih2o/amass_phc_filtered.pkl"
        self.motions.hoverh1.step_dt = self.decimation * self.sim.dt
        self.motions.hoverh1.random_sample = True


@configclass
class BOTH1Cfg_PLAY(BOTH1Cfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 16
        # self.scene.env_spacing = 2.5

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
        # self.events.interval_push = None
        # self.events.interval_gravity = None
        # self.events.interval_actuator = None
        # self.events.interval_mass = None
        # self.events.interval_coms = None

        # self.motions.hoverh1.random_sample = False
        self.motions.hoverh1.debug_vis = True
        self.motions.hoverh1.random_sample = False
        self.motions.hoverh1.resample_interval_s = 15


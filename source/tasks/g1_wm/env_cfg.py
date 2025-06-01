from isaaclabex.assets.robots import unitree_g129
from isaaclabex.scenes import scenes_cfg
from isaaclabex.envs import rl_env_exts_cfg
from isaaclab.utils import configclass
from .mdps  import mdps, curriculum, rewards, obs
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

@configclass
class G1RoughEnvCfg(rl_env_exts_cfg.ManagerBasedRLExtendsCfg):
    # Scene settings
    scene = scenes_cfg.BaseSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: obs.ObservationsCfg = obs.ObservationsCfg()
    actions =  mdps.ActionsCfg()
    commands = mdps.CommandsCfg()
    # MDP settings
    rewards = rewards.RewardsCfg()
    terminations = mdps.TerminationsCfg()
    events = mdps.EventCfg()
    curriculum = curriculum.CurriculumCfg()

    def __post_init__(self):
        # ROBOT
        self.scene.robot = unitree_g129.UNITREE_GO129_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
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
class G1NoWMEnvCfg(G1RoughEnvCfg):
    # reward positive flag
    #reward_positive_flag = True

    def __post_init__(self):
        super(G1NoWMEnvCfg, self).__post_init__()
        self.observations.policy.concatenate_terms = True
        self.observations.policy.ang_vel.noise = Unoise(n_min=-0.2, n_max=0.2)
        self.observations.policy.gravity.noise = Unoise(n_min=-0.05, n_max=0.05)
        #self.observations.policy.commands.noise = Unoise(n_min=-0.1, n_max=0.1)
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.01, n_max=0.01)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-1.5, n_max=1.5)
        #self.observations.policy.actions.noise = Unoise(n_min=-0.1, n_max=0.1)
        self.observations.policy.lin_vel.noise = Unoise(n_min=-0.1, n_max=0.1)
        self.observations.policy.contact_forces.noise = Unoise(n_min=-0.2, n_max=0.2)
        self.observations.policy.payload.noise = Unoise(n_min=-0.2, n_max=0.2)
        self.observations.policy.stiffness.noise = Unoise(n_min=-0.2, n_max=0.2)
        self.observations.policy.damping.noise = Unoise(n_min=-0.2, n_max=0.2)

@configclass
class G1NoWMEnvCfg_PLAY(G1NoWMEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
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
        self.events = None
        self.curriculum = None


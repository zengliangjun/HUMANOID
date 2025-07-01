from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg, ObservationGroupCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclabex.envs.mdp.observations import privileged, ext_obs


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""
        phase = ObservationTermCfg(func=ext_obs.phase_commands, params={"command_name": "phase_command"}, clip =(-18, 18))
        commands = ObservationTermCfg(func=mdp.generated_commands, scale = 0.25, params={"command_name": "base_velocity"}, clip = (-18, 18))
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.025, n_max=0.025), clip = (-18, 18))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, scale = 0.05, noise=Unoise(n_min=-0.25, n_max=0.25), clip = (-18, 18))

        actions = ObservationTermCfg(func=mdp.last_action, clip = (-18, 18))
        ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, scale = 1, noise=Unoise(n_min=-0.05, n_max=0.05), clip = (-18, 18))
        euler = ObservationTermCfg(func=ext_obs.root_euler_w, scale = 0.25, noise=Unoise(n_min=-0.015, n_max=0.015), clip = (-18, 18))

        history_length = 15

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        ## proprioceptive
        lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, scale = 2.0, noise=Unoise(n_min=-0.025, n_max=0.025), clip = (-18, 18))
        push_force = ObservationTermCfg(
            func=privileged.push_force,
            params={
                "asset_cfg": SceneEntityCfg("robot")
            }, clip = (-18, 18)
        )
        push_torque = ObservationTermCfg(
            func=privileged.push_torque,
            params={
                "asset_cfg": SceneEntityCfg("robot")
            }, clip = (-18, 18)
        )
        frictions = ObservationTermCfg(
            func=privileged.joint_friction_coeff,
            params={
                "asset_cfg": SceneEntityCfg("robot")
            }, clip = (-18, 18)
        )
        body_mass = ObservationTermCfg(
            func=privileged.body_mass, scale = 1 / 30,
            params={
                "asset_cfg": SceneEntityCfg("robot")
            }, clip = (-18, 18)
        )

        def __post_init__(self):
            self.history_length = 3

            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

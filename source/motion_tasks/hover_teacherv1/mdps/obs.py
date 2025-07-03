from isaaclab.utils import configclass
from dataclasses import MISSING

from isaaclab.managers import ObservationTermCfg, ObservationGroupCfg, SceneEntityCfg

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclabmotion.envs.mdp.observations import body, motions
from isaaclabex.envs.mdp.observations import privileged
import copy

bodyparams={
    "asset_cfg": SceneEntityCfg("robot",
        body_names= [
            'left_hip_yaw_link',
            'left_hip_roll_link',
            'left_hip_pitch_link',
            'left_knee_link',
            'left_ankle_link',

            'right_hip_yaw_link',
            'right_hip_roll_link',
            'right_hip_pitch_link',
            'right_knee_link',
            'right_ankle_link',

            'torso_link',

            'left_shoulder_pitch_link',
            'left_shoulder_roll_link',
            'left_shoulder_yaw_link',
            'left_elbow_link',
            'right_shoulder_pitch_link',
            'right_shoulder_roll_link',
            'right_shoulder_yaw_link',
            'right_elbow_link'
        ], preserve_order = True),
    "motions_name": "hoverh1",
    "extend_body_names": [
        "left_hand_link",
        "right_hand_link",
        "head_link"
    ]
}

mtparams = copy.deepcopy(bodyparams)
setattr(mtparams["asset_cfg"], "body_names", mtparams["asset_cfg"].body_names + ['pelvis'])

rb_pos_noise_scale = Unoise(n_min=-0.01, n_max=0.01)
rb_rot_noise_scale = Unoise(n_min=-0.01, n_max=0.01)
rb_lin_noise_scale = Unoise(n_min=-0.01, n_max=0.01)
rb_ang_noise_scale = Unoise(n_min=-0.01, n_max=0.01)
mdiff_rbpos_noise_scale = Unoise(n_min=-0.01, n_max=0.01)
mdiff_rbquat_noise_scale = Unoise(n_min=-0.01, n_max=0.01)
mdiff_rblin_noise_scale = Unoise(n_min=-0.01, n_max=0.01)
mdiff_rbang_noise_scale = Unoise(n_min=-0.01, n_max=0.01)
mdiff_root_rbpos_noise_scale = Unoise(n_min=-0.01, n_max=0.01)
mdiff_root_rbquat_noise_scale = Unoise(n_min=-0.01, n_max=0.01)
m_rbpos_noise_scale = Unoise(n_min=-0.01, n_max=0.01)
m_rbquat_noise_scale = Unoise(n_min=-0.01, n_max=0.01)
actions_noise_scale = Unoise(n_min=-0.01, n_max=0.01)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        rb_pos = ObservationTermCfg(
            func=body.obs_body_pos,
            params = bodyparams,
            scale = 10, noise=rb_pos_noise_scale)

        rb_rot = ObservationTermCfg(
            func=body.obs_body_rotwxyz,
            params = mtparams,
            scale = 10, noise=rb_rot_noise_scale)

        rb_lin = ObservationTermCfg(
            func=body.obs_body_lin_vel,
            params = mtparams,
            scale = 5, noise=rb_lin_noise_scale)

        rb_ang = ObservationTermCfg(
            func=body.obs_body_ang_vel,
            params = mtparams,
            scale = 5, noise=rb_ang_noise_scale)

        # motion
        mdiff_rbpos = ObservationTermCfg(
            func=motions.obs_diff_rbpos,
            params = mtparams,
            scale = 10, noise=mdiff_rbpos_noise_scale)

        mdiff_rbquat = ObservationTermCfg(
            func=motions.obs_diff_rbquat,
            params = mtparams,
            scale = 10, noise=mdiff_rbquat_noise_scale)

        mdiff_rblin = ObservationTermCfg(
            func=motions.obs_diff_rblin,
            params = mtparams,
            scale = 5, noise=mdiff_rblin_noise_scale)

        mdiff_rbang = ObservationTermCfg(
            func=motions.obs_diff_rbang,
            params = mtparams,
            scale = 5, noise=mdiff_rbang_noise_scale)
        # root
        """
        mdiff_root_rbpos = ObservationTermCfg(
            func=motions.obs_diff_root_rbpos,
            params = mtparams,
            scale = 1.2, noise=mdiff_root_rbpos_noise_scale)

        mdiff_root_rbquat = ObservationTermCfg(
            func=motions.obs_diff_root_rbquat,
            params = mtparams,
            scale = 1.2, noise=mdiff_root_rbquat_noise_scale)
        """
        # motions

        m_rbpos = ObservationTermCfg(
            func=motions.obs_motions_rbpos,
            params = mtparams,
            scale = 10, noise=m_rbpos_noise_scale)

        m_rbquat = ObservationTermCfg(
            func=motions.obs_motions_rbquat,
            params = mtparams,
            scale = 5, noise=m_rbquat_noise_scale)

        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        p_coms = ObservationTermCfg(
            func=privileged.body_coms,
            params={"asset_cfg": SceneEntityCfg("robot", body_names =["pelvis"])},
            scale = 1)

        p_frictions = ObservationTermCfg(
            func=privileged.joint_friction_coeff,
            params = {
                "asset_cfg": SceneEntityCfg("robot",
                                joint_names = [
                                    'left_ankle',
                                    'right_ankle',
                                ])
            },
            scale = 1)

        p_stiffness = ObservationTermCfg(
            func=privileged.joint_stiffness,
            params = {
                "asset_cfg": SceneEntityCfg("robot")
            },
            scale = 0.08)

        p_damping = ObservationTermCfg(
            func=privileged.joint_damping,
            params = {
                "asset_cfg": SceneEntityCfg("robot")
            },
            scale = 0.25)

        p_forces = ObservationTermCfg(
            func=privileged.feet_contact_forces,
            params = {
                "sensor_cfg": SceneEntityCfg("contact_forces",
                                body_names= [
                                    'left_ankle_link',
                                    'right_ankle_link',
                                    ])
            },
            scale = 0.05)

        p_mass = ObservationTermCfg(
            func=privileged.body_mass,
            params = {
                "asset_cfg": SceneEntityCfg("robot",
                                body_names= [
                                    'pelvis',

                                    'left_hip_yaw_link',
                                    'left_hip_roll_link',
                                    'left_hip_pitch_link',
                                    'left_knee_link',
                                    'left_ankle_link',

                                    'right_hip_yaw_link',
                                    'right_hip_roll_link',
                                    'right_hip_pitch_link',
                                    'right_knee_link',
                                    'right_ankle_link',

                                    'torso_link',

                                    'left_shoulder_pitch_link',
                                    'left_shoulder_roll_link',
                                    'left_shoulder_yaw_link',
                                    'left_elbow_link',
                                    'right_shoulder_pitch_link',
                                    'right_shoulder_roll_link',
                                    'right_shoulder_yaw_link',
                                    'right_elbow_link'
                                ])
            },
            scale = 1)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

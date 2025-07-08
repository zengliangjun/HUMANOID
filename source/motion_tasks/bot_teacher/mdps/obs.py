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
            'right_hip_yaw_link',

            'torso_link',

            'left_hip_roll_link',
            'right_hip_roll_link',
            'left_shoulder_pitch_link',
            'right_shoulder_pitch_link',

            'left_hip_pitch_link',
            'right_hip_pitch_link',
            'left_shoulder_roll_link',
            'right_shoulder_roll_link',

            'left_knee_link',
            'left_ankle_link',
            'left_shoulder_yaw_link',
            'right_shoulder_yaw_link',

            'right_knee_link',
            'right_ankle_link',
            'left_elbow_link',
            'right_elbow_link'
        ], preserve_order = True),

    "motions_name": "hoverh1",

    "extend_body_names": [
        "head_link",
        "left_hand_link",
        "right_hand_link"
    ]
}

mtparams = copy.deepcopy(bodyparams)
setattr(mtparams["asset_cfg"], "body_names", ['pelvis'] + mtparams["asset_cfg"].body_names)

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

        p_mass = ObservationTermCfg(
            func=privileged.body_mass,
            params = {
                "asset_cfg": mtparams["asset_cfg"]
            },
            scale = 1)

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


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


from rsl_rlex.bot.modules import mappings

token_names = [
    'root',
    'left_hip_yaw',
    'right_hip_yaw',
    'torso',

    'left_hip_roll',
    'right_hip_roll',
    'left_shoulder_pitch',
    'right_shoulder_pitch',

    'left_hip_pitch',
    'right_hip_pitch',
    'left_shoulder_roll',
    'right_shoulder_roll',

    'left_knee',
    'right_knee',
    'left_shoulder_yaw',
    'right_shoulder_yaw',

    'left_ankle',
    'right_ankle',
    'left_elbow',
    'right_elbow',
]

def build_ids(slices_array, start_slicesid, start_tokenid, token_count, stride, extends_ids):

    for tokenid in range(start_tokenid, token_count):
        slicesids = [start_slicesid + i for i in range(stride)]
        slices_array[tokenid].extend(slicesids)

        start_slicesid += stride

    for extend_id in extends_ids:
        slicesids = [start_slicesid + i for i in range(stride)]
        slices_array[extend_id].extend(slicesids)

        start_slicesid += stride

    return start_slicesid


def build_actor_inputslices():
    inputslices = [[] for id in range(len(token_names))]
    start_slicesid = 0
    start_slicesid = build_ids(inputslices, start_slicesid, 1, len(token_names), 3, [0, 18, 19])  # for rb_pos
    print("rb_pos:\n", inputslices)
    start_slicesid = build_ids(inputslices, start_slicesid, 0, len(token_names), 6, [0, 18, 19])  # for rb_rot
    print("rb_rot:\n", inputslices)
    start_slicesid = build_ids(inputslices, start_slicesid, 0, len(token_names), 3, [0, 18, 19])  # for rb_lin
    print("rb_lin:\n", inputslices)
    start_slicesid = build_ids(inputslices, start_slicesid, 0, len(token_names), 3, [0, 18, 19])  # for rb_ang
    print("rb_ang:\n", inputslices)
    start_slicesid = build_ids(inputslices, start_slicesid, 0, len(token_names), 3, [0, 18, 19])  # for mdiff_rbpos
    print("mdiff_rbpos:\n", inputslices)
    start_slicesid = build_ids(inputslices, start_slicesid, 0, len(token_names), 6, [0, 18, 19])  # for mdiff_rbquat
    print("mdiff_rbquat:\n", inputslices)
    start_slicesid = build_ids(inputslices, start_slicesid, 0, len(token_names), 3, [0, 18, 19])  # for mdiff_rblin
    print("mdiff_rblin:\n", inputslices)
    start_slicesid = build_ids(inputslices, start_slicesid, 0, len(token_names), 3, [0, 18, 19])  # for mdiff_rbang
    print("mdiff_rbang:\n", inputslices)
    start_slicesid = build_ids(inputslices, start_slicesid, 0, len(token_names), 3, [0, 18, 19])  # for m_rbpos
    print("m_rbpos:\n", inputslices)
    start_slicesid = build_ids(inputslices, start_slicesid, 0, len(token_names), 6, [0, 18, 19])  # for m_rbquat
    print("m_rbquat:\n", inputslices)
    start_slicesid = build_ids(inputslices, start_slicesid, 1, len(token_names), 1, [])  # for actions
    print("m_rbquat:\n", inputslices)
    return inputslices, start_slicesid

def build_critic_inputslices():
    inputslices, start_slicesid = build_actor_inputslices()

    start_slicesid = build_ids(inputslices, start_slicesid, 1, len(token_names), 1, [])  # for p_stiffness
    print("m_rbquat:\n", inputslices)
    start_slicesid = build_ids(inputslices, start_slicesid, 1, len(token_names), 1, [])  # for p_damping
    print("p_damping:\n", inputslices)
    start_slicesid = build_ids(inputslices, start_slicesid, 0, len(token_names), 1, [])  # for p_mass
    print("p_damping:\n", inputslices)

    start_slicesid = build_ids(inputslices, start_slicesid, 0, 1, 3, [])  # for p_coms
    print("p_coms:\n", inputslices)

    start_slicesid = build_ids(inputslices, start_slicesid, 0, 0, 1, [13, 17])  # for p_frictions
    print("p_coms:\n", inputslices)

    start_slicesid = build_ids(inputslices, start_slicesid, 0, 0, 3, [13, 17])  # for p_forces
    print("p_coms:\n", inputslices)
    return inputslices, start_slicesid


def build_actor_outputslices():
    inputslices = [[] for id in range(len(token_names))]
    start_slicesid = 0
    start_slicesid = build_ids(inputslices, start_slicesid, 1, len(token_names), 1, [])  # for rb_pos
    return inputslices, start_slicesid

def build_critic_outputslices():
    inputslices = [[] for id in range(len(token_names))]
    start_slicesid = 0
    start_slicesid = build_ids(inputslices, start_slicesid, 1, len(token_names), 1, [])  # for rb_pos
    return inputslices, start_slicesid

actor_mapping = mappings.token_mapping(
    token_names,
    build_actor_inputslices()[0],
    build_actor_outputslices()[0])

critic_mapping = mappings.token_mapping(
    token_names,
    build_critic_inputslices()[0],
    build_critic_outputslices()[0])



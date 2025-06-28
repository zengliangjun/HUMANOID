from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg
from isaaclabex.envs.mdp.curriculum import events, adaptive
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class CurriculumCfg:

    terrain_levels = CurriculumTermCfg(func=mdp.terrain_levels_vel)

    events_with_steps = CurriculumTermCfg(
        func=events.range_with_degree,
        params={
            "degree": 0.000015,
            "down_up_lengths":[400, 750],
            "scale_range": [0, 1],
            "manager_name": "event",
            "curriculums": {
                'startup_material': {    # event name
                    "static_friction_range": events.EventCurriculumStepItem(
                        start_range = (1, 1),
                        end_range = (0.05, 1.5)
                    ),
                    "dynamic_friction_range": events.EventCurriculumStepItem(
                        start_range = (1, 1),
                        end_range = (0.8, 1.2)
                    )
                },
                'reset_base': {    # event name
                    "pose_range": events.EventCurriculumStepItem(
                        start_range = {"x": (-0, 0), "y": (-0, 0), "yaw": (-0, 0)},
                        end_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}
                    ),
                    "velocity_range": events.EventCurriculumStepItem(
                        start_range =  {
                                            "x": (-0.0, 0.),
                                            "y": (-0.0, 0.0),
                                            "z": (-0.0, 0.0),
                                            "roll": (-0., 0.0),
                                            "pitch": (-0.0, 0.0),
                                            "yaw": (-0.0, 0.0),
                                        },
                        end_range =  {
                                        "x": (-0.2, 0.2),
                                        "y": (-0.2, 0.2),
                                        "z": (-0.2, 0.2),
                                        "roll": (-0.2, 0.2),
                                        "pitch": (-0.2, 0.2),
                                        "yaw": (-0.2, 0.2),
                                    }
                    ),
                },
                'reset_joints': {    # event name
                    "position_range": events.EventCurriculumStepItem(
                        start_range = (0, 0),
                        end_range = (0.5, 1.5)
                    ),
                },
                'interval_push': {    # event name
                    "velocity_range": events.EventCurriculumStepItem(
                        start_range =  {"x": (-0.0, 0.0), "y": (-0.0, 0.0)},
                        end_range =  {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}
                    ),
                },
                'interval_gravity': {    # event name
                    "gravity_distribution_params": events.EventCurriculumStepItem(
                        start_range =  (0, 0),
                        end_range =  (-0.1, 0.1)
                    ),
                },
                'interval_actuator': {    # event name
                    "stiffness_distribution_params": events.EventCurriculumStepItem(
                        start_range =  (1, 1),
                        end_range =  (.8, 1.2)
                    ),
                    "damping_distribution_params": events.EventCurriculumStepItem(
                        start_range =  (1, 1),
                        end_range =  (.8, 1.2)
                    ),
                },
                'interval_mass': {    # event name
                    "mass_distribution_params": events.EventCurriculumStepItem(
                        start_range = (1, 1),
                        end_range = (0.85, 1.15)
                    ),
                },
                'interval_coms': {    # event name
                    "coms_distribution_params": events.EventCurriculumStepItem(
                        start_range = (0, 0),
                        end_range = (-0.15, 0.15)
                    ),
                },
            }
        },
    )

    p_reward_steps = CurriculumTermCfg(
        func=adaptive.scale_with_degree,
        params={
            "degree": 0.000015,
            "down_up_lengths":[400, 750],
            "scale_range": [0, 1],
            "manager_name": "reward",
            "curriculums": {
                'p_action_rate': {    # reward name
                    "param_name": "weight",
                    "start_weight": -0.01,
                    "end_weight": -0.1
                },
                'p_action_smoothness': {    # reward name
                    "param_name": "weight",
                    "start_weight": -0.001,
                    "end_weight": -0.02
                },
                'p_torques': {    # reward name
                    "param_name": "weight",
                    "start_weight": -0.0001,
                    "end_weight": -0.001
                },
                'p_width': {    # reward name
                    "param_name": "weight",
                    "start_weight": -3.0,
                    "end_weight": -10
                },
                'p_orientation': {    # reward name
                    "param_name": "weight",
                    "start_weight": -1.0,
                    "end_weight": -10
                },
                'p_height': {    # reward name
                    "param_name": "weight",
                    "start_weight": -10.0,
                    "end_weight": -40.0
                },
                'p_foot_clearance': {    # reward name
                    "param_name": "weight",
                    "start_weight": -20.0,
                    "end_weight": -80.0
                }
            }
        }
    )

    rewm_ankler_z = CurriculumTermCfg(
        func=adaptive.curriculum_with_degree,
        params={
            "degree": - 0.00003,
            "down_up_lengths": [600, 850],
            "value_range": [0.08, 0.10 * 1.6],
            "manager_name": "reward",
            "term_name": "rew_mean_ankler_zero",
            "param_name": "std"
        }
    )
    rewm_hipr_z = CurriculumTermCfg(
        func=adaptive.curriculum_with_degree,
        params={
            "degree": - 0.00003,
            "down_up_lengths": [600, 850],
            "value_range": [0.06, 0.06 * 2],
            "manager_name": "reward",
            "term_name": "rew_mean_hipr_zero",
            "param_name": "std"
        }
    )
    rewm_hipy_z = CurriculumTermCfg(
        func=adaptive.curriculum_with_degree,
        params={
            "degree": - 0.00003,
            "down_up_lengths": [600, 850],
            "value_range": [0.03, 0.06 * 2],
            "manager_name": "reward",
            "term_name": "rew_mean_hipy_zero",
            "param_name": "std"
        }
    )
    """
    r_hipp = CurriculumTermCfg(
        func=adaptive.curriculum_with_degree,
        params={
            "degree": 0.00004,
            "down_up_lengths": [600, 850],
            "value_range": [0.04, 0.09 * 1.5],
            "manager_name": "reward",
            "term_name": "rew_hipp_self",
            "param_name": "std"
        }
    )
    r_knee = CurriculumTermCfg(
        func=adaptive.curriculum_with_degree,
        params={
            "degree": 0.00004,
            "down_up_lengths": [600, 900],
            "value_range": [0.04, 0.09 * 1.5],
            "manager_name": "reward",
            "term_name": "rew_knee_self",
            "param_name": "std"
        }
    )
    """
    r_ankler_z = CurriculumTermCfg(
        func=adaptive.curriculum_with_degree,
        params={
            "degree": - 0.00003,
            "down_up_lengths": [600, 850],
            "value_range": [0.006, 0.01 * 2.5],
            "manager_name": "reward",
            "term_name": "rew_ankler_zero",
            "param_name": "std"
        }
    )
    r_hipr_z = CurriculumTermCfg(
        func=adaptive.curriculum_with_degree,
        params={
            "degree": - 0.00003,
            "down_up_lengths": [600, 850],
            "value_range": [0.003, 0.008 * 2.5],
            "manager_name": "reward",
            "term_name": "rew_hipr_zero",
            "param_name": "std"
        }
    )
    r_hipy_z = CurriculumTermCfg(
        func=adaptive.curriculum_with_degree,
        params={
            "degree": - 0.00003,
            "down_up_lengths": [600, 850],
            "value_range": [0.003, 0.008 * 3],
            "manager_name": "reward",
            "term_name": "rew_hipy_zero",
            "param_name": "std"
        }
    )

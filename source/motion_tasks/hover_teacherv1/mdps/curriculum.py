from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg
from isaaclabex.envs.mdp.curriculum import events, adaptive
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class CurriculumCfg:

    events_with_steps = CurriculumTermCfg(
        func=events.range_with_degree,
        params={
            "degree": 0.00001,
            "down_up_lengths":[35, 45],
            "scale": 0.0,
            "scale_range": [0.0, 1],
            "manager_name": "event",
            "curriculums": {
                'startup_material': {    # event name
                    "static_friction_range": events.EventCurriculumStepItem(
                        start_range = (1, 1),
                        end_range = (0.7, 1.3)
                    ),
                },
                'interval_push': {    # event name
                    "velocity_range": events.EventCurriculumStepItem(
                        start_range =  {"x": (-0.0, 0.0), "y": (-0.0, 0.0)},
                        end_range =  {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}
                    ),
                },
                'interval_mass': {    # event name
                    "mass_distribution_params": events.EventCurriculumStepItem(
                        start_range = (1, 1),
                        end_range = (0.7, 1.3)
                    ),
                },
                'interval_coms': {    # event name
                    "coms_distribution_params": events.EventCurriculumStepItem(
                        start_range = (0, 0),
                        end_range = (-0.1, 0.1)
                    ),
                },
                'reset_actuator': {    # event name
                    "stiffness_distribution_params": events.EventCurriculumStepItem(
                        start_range =  (1, 1),
                        end_range =  (0.75, 1.25)
                    ),
                    "damping_distribution_params": events.EventCurriculumStepItem(
                        start_range =  (1, 1),
                        end_range =  (0.75, 1.25)
                    ),
                },
            }
        },
    )

    p_terminations_steps = CurriculumTermCfg(
        func=adaptive.scale_with_degree,
        params={
            "degree": 0.000001,
            "down_up_lengths":[35, 40],
            "scale": 0.5,
            "scale_range": [0.18, 1],
            "manager_name": "termination",
            "curriculums": {
                'distance': {    # reward name
                    "param_name": "max_ref_motion_dist",
                    "start_max_ref_motion_dist": 0.8,
                    "end_max_ref_motion_dist": 0.3
                }
            }
        }
    )

    p_reward_steps = CurriculumTermCfg(
        func=adaptive.scale_with_degree,
        params={
            "degree": 0.00001,
            "down_up_lengths":[30, 40],
            "scale": 0.5,
            "scale_range": [0.18, 1],
            "manager_name": "reward",
            "curriculums": {
                'p_torques': {    # reward name
                    "param_name": "weight",
                    "start_weight": -0,
                    "end_weight": -0.0001
                },
                'p_torque_limits': {    # reward name
                    "param_name": "weight",
                    "start_weight": -0,
                    "end_weight": -2
                },
                'p_jacc': {    # reward name
                    "param_name": "weight",
                    "start_weight": 0,
                    "end_weight": -0.000011
                },
                'p_jvel': {    # reward name
                    "param_name": "weight",
                    "start_weight": 0,
                    "end_weight": -0.004
                },
                'p_lower_actionrate': {    # reward name
                    "param_name": "weight",
                    "start_weight": 0,
                    "end_weight": -3.0
                },
                'p_upper_actionrate': {    # reward name
                    "param_name": "weight",
                    "start_weight": 0,
                    "end_weight": -0.625
                },
                'p_jpos_limits': {    # reward name
                    "param_name": "weight",
                    "start_weight": 0,
                    "end_weight": -125.0
                },
                'p_jvel_limits': {    # reward name
                    "param_name": "weight",
                    "start_weight": 0,
                    "end_weight": -50.0
                },
                'p_termination': {    # reward name
                    "param_name": "weight",
                    "start_weight": 0,
                    "end_weight": -250
                },
                'p_cforces': {    # reward name
                    "param_name": "weight",
                    "start_weight": 0,
                    "end_weight": -0.75
                },
                'p_stumble': {    # reward name
                    "param_name": "weight",
                    "start_weight": 0,
                    "end_weight": -1000.0
                },
                'p_slippage': {    # reward name
                    "param_name": "weight",
                    "start_weight": 0,
                    "end_weight": -37.5
                },
                'p_feet_ori': {    # reward name
                    "param_name": "weight",
                    "start_weight": 0,
                    "end_weight": -62.5
                },
                'p_both_air': {    # reward name
                    "param_name": "weight",
                    "start_weight": 0,
                    "end_weight": -200.0
                },
                'p_feet_height': {    # reward name
                    "param_name": "weight",
                    "start_weight": 0,
                    "end_weight": -2500.0
                }
            }
        }
    )

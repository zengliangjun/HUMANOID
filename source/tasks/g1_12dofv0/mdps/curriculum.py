from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg
from isaaclabex.envs.mdp.curriculum import events, rewards
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class CurriculumCfg:

    terrain_levels = CurriculumTermCfg(func=mdp.terrain_levels_vel)

    events_with_steps = CurriculumTermCfg(
        func=events.curriculum_with_steps,
        params={
            'start_steps': 0,
            'end_steps': 1600000,
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

    penalize_with_steps = CurriculumTermCfg(
        func=rewards.curriculum_with_steps,
        params={
            'start_steps': 0,
            'end_steps': 1600000,
            "curriculums": {
                'p_action_rate': {    # reward name
                    "start_weight": -0.02,
                    "end_weight": -0.1
                },
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
                },
                'p_foot_clearance': {    # reward name
                    "start_weight": -20.0,
                    "end_weight": -80.0
                }
            }
        }
    )

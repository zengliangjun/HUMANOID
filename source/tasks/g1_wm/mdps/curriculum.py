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
            'start_steps': 2000000,
            'end_steps': 8000000,
            "curriculums": {
                'startup_material': {    # event name
                    "static_friction_range": events.EventCurriculumStepItem(
                        start_range = (1, 1),
                        end_range = (0.6, 1.4)
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
                                        "x": (-0.5, 0.5),
                                        "y": (-0.5, 0.5),
                                        "z": (-0.5, 0.5),
                                        "roll": (-0.5, 0.5),
                                        "pitch": (-0.5, 0.5),
                                        "yaw": (-0.5, 0.5),
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
                        end_range =  {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
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
                        end_range = (0.9, 1.1)
                    ),
                },
            }
        },
    )

    rewards_with_steps = CurriculumTermCfg(
        func=rewards.curriculum_with_steps,
        params={
            'start_steps': 0,
            'end_steps': 2000000,
            "curriculums": {
                'rew_tracking_linear': {    # reward name
                    "start_weight": 8,
                    "end_weight": 1
                },
                'rew_tracking_z': {    # reward name
                    "start_weight": 4,
                    "end_weight": 1
                },
                'rew_air_time': {    # reward name
                    "start_weight": 0.05,
                    "end_weight": 0.2
                },
                'rew_feet_forces_z': {    # reward name
                    "start_weight": 5e-4,
                    "end_weight": 5e-3
                },
                'penalize_linear_xy': {    # reward name
                    "start_weight": -0.02,
                    "end_weight": -0.05
                },
                'penalize_linear_z': {    # reward name
                    "start_weight": -0.2,
                    "end_weight": -1
                },
                'penalize_orientation': {    # reward name
                    "start_weight": -0.5,
                    "end_weight": -2
                },
                'penalize_energy': {    # reward name
                    "start_weight": -0.0002,
                    "end_weight": -0.001
                },
                'penalize_joint_acc': {    # reward name
                    "start_weight": -2.5e-8,
                    "end_weight": -2.5e-7
                },
                'penalize_pos_limits': {    # reward name
                    "start_weight": -0.5,
                    "end_weight": -2.0
                },
                'penalize_pitch': {    # reward name
                    "start_weight": -.005,
                    "end_weight": -.025
                },
                'penalize_hip': {    # reward name
                    "start_weight": -.05,
                    "end_weight": -.1
                },
                'penalize_other': {    # reward name
                    "start_weight": -0.001,
                    "end_weight": -0.05
                },
                'penalize_waist': {    # reward name
                    "start_weight": -0.08,
                    "end_weight": -0.3
                },
                'penalize_feet_stumble': {    # reward name
                    "start_weight": 1e-3,
                    "end_weight": 5e-3
                },
                'penalize_foot_slide': {    # reward name
                    "start_weight": -0.1,
                    "end_weight": -0.5
                },
                'penalize_feet_airborne': {    # reward name
                    "start_weight": -.05,
                    "end_weight": -.1
                },

                'penalize_action_rate': {    # reward name
                    "start_weight": -0.001,
                    "end_weight": -0.01
                },

            }
        }
    )

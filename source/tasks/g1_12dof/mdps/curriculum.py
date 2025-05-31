from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg
from isaaclabex.envs.mdp.curriculum import events
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class CurriculumCfg:

    events_with_steps = CurriculumTermCfg(
        func=events.curriculum_with_steps,
        params={
            'start_steps': 0,
            'end_steps': 800000,
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
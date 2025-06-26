from isaaclab.envs.mdp import rewards as isaaclab_rewards
from isaaclab_tasks.manager_based.classic.cartpole.mdp import rewards as cartpole_rewards
from isaaclab_tasks.manager_based.classic.humanoid.mdp import rewards as humanoid_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import rewards as loc_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp import rewards as spot_rewards
from .joint import joint, phase, statistics, symmetry, statisticsv2_pos, statisticsv3_pos
from .feet import feet_phase, feet, feet_contact, statistics as feet_statistics, statisticsv3 as feet_statisticsv3
from . import actions
from .root import base, base_ori, base_phase, body, body_status, contact

'''
Episodic Rewards:
These parameters reward or penalize the high-level episode state,
such as whether the agent is still alive or if the simulation has terminated.
'''
rewards_eps_alive = isaaclab_rewards.is_alive                       # Reward for being alive.
penalize_eps_terminated = isaaclab_rewards.is_terminated             # Penalty when terminated abnormally.
penalize_eps_terminated_term = isaaclab_rewards.is_terminated_term     # Penalty when terminated in terminal state.

"""
Base Rewards and Penalties:
Reward and penalty values pertaining to the base's orientation, height, and velocity tracking.
"""
# Orientation rewards/penalties: penalize orientation deviation measured by different metrics.
penalize_ori_l2 = isaaclab_rewards.flat_orientation_l2                # L2 penalty for flat orientation error.
penalize_ori_norm = spot_rewards.base_orientation_penalty             # Norm penalty for orientation error.
reward_ori_euler_gravity_b = base_ori.reward_orientation_euler_gravity_b  # Euler-based orientation reward.
reward_ori_gravity = humanoid_rewards.upright_posture_bonus            # Bonus for maintaining upright gravity alignment.

# Height-based rewards/penalties.
penalize_height_flat_or_rayl2 = isaaclab_rewards.base_height_l2         # L2 penalty on base height error (flat/ray).
penalize_height_base2feet = base.penalize_base_height                   # Penalty for mismatch in base-to-feet height.
reward_height_base2feet_phase = base_phase.reward_base_height           # Reward based on phase consistency between base and feet height.

# Velocity tracking rewards/penalties.
penalize_lin_z_l2 = isaaclab_rewards.lin_vel_z_l2                      # L2 penalty for linear velocity (z).
penalize_ang_xy_l2 = isaaclab_rewards.ang_vel_xy_l2                    # L2 penalty for angular velocity (xy).
penalize_motion_lin_ang = spot_rewards.base_motion_penalty              # Penalty for error in combined linear and angular velocities.
reward_motion_lin_ang = base.reward_mismatch_vel_exp                    # Exponential penalty for velocity mismatch.
reward_motion_speed = base.reward_mismatch_speed                       # Reward based on matching speed.
reward_motion_hard = base.reward_track_vel_hard                        # Hard matching reward for velocity tracking.

# Additional velocity tracking rewards.
reward_lin_xy_exp = isaaclab_rewards.track_lin_vel_xy_exp                # Exponential reward for linear xy velocity tracking.
reward_ang_z_exp = isaaclab_rewards.track_ang_vel_z_exp                  # Exponential reward for angular z velocity tracking.
reward_lin_xy_exp2 = loc_rewards.track_lin_vel_xy_yaw_frame_exp           # Alternative linear velocity reward in yaw frame.
reward_ang_z_exp2 = loc_rewards.track_ang_vel_z_world_exp                 # Alternative angular velocity reward in world frame.
reward_lin_xy_exp3 = spot_rewards.base_linear_velocity_reward            # Additional linear velocity reward sample.
reward_ang_z_exp3 = spot_rewards.base_angular_velocity_reward             # Additional angular velocity reward sample.

# Target position rewards.
reward_move_to_target_bonus = humanoid_rewards.move_to_target_bonus        # Bonus for reaching target.
reward_progress_reward = humanoid_rewards.progress_reward                  # Reward based on progression.

# Base acceleration regularization.
reward_base_acc_exp_norm = base.reward_base_acc                           # Reward to regularize base acceleration.

"""
Body Rewards:
Penalties and rewards specific to the body dynamics.
"""
penalize_body_lin_acc_l2 = isaaclab_rewards.body_lin_acc_l2               # L2 penalty on body linear acceleration.
reward_body_distance = body.reward_distance                              # Reward based on body distance metric.
reward_width = body.reward_width                                 # Reward based on body width metric.
penalize_width = body.penalize_width

reward_stability = body_status.Stability
"""
Joint Penalties:
Penalties for energy consumption, torque, joint positions, velocities, and acceleration.
"""
# Energy and torque related penalties.
penalize_torques_l2 = isaaclab_rewards.joint_torques_l2                   # L2 penalty for joint torques.
penalize_torques_norm = spot_rewards.joint_torques_penalty                # Norm penalty for joint torques.
penalize_action_vel = humanoid_rewards.power_consumption                # Penalty on power consumption (action x velocity).
penalize_energy = joint.energy_cost                                     # Energy cost penalty from joint actions.
penalize_torque_limits = isaaclab_rewards.applied_torque_limits           # Penalty for exceeding torque limits.

# Joint position constraints.
penalize_jpos_limits_l1 = isaaclab_rewards.joint_pos_limits               # L1 penalty for joint position limits.
try:
    penalize_jpos_limits_ratio = humanoid_rewards.joint_pos_limits_penalty_ratio  # Preferred joint limits penalty ratio.
except:
    penalize_jpos_limits_ratio = humanoid_rewards.joint_limits_penalty_ratio        # Fallback joint limits penalty ratio.
# Joint position regularization.
penalize_jpos_deviation_l1 = isaaclab_rewards.joint_deviation_l1           # L1 penalty for deviation from desired joint positions.
penalize_jpos_norm_stand_check = spot_rewards.joint_position_penalty        # Norm penalty for joint positions.
reward_penalize_joint = joint.reward_penalize_joint            # Reward for correct yaw/roll in joint positions.
reward_jpos_withrefpose = phase.rew_joint_pos_withrefpose                  # Reward for matching a reference pose.

# Target joint positions.
penalize_jpos_target_l2 = cartpole_rewards.joint_pos_target_l2             # L2 penalty to drive joint positions toward targets.

# Joint velocity penalties.
penalize_jvel_l1 = isaaclab_rewards.joint_vel_l1                          # L1 penalty for joint velocities.
penalize_jvel_l2 = isaaclab_rewards.joint_vel_l2                          # L2 penalty for joint velocities.
penalize_jvel_limits = isaaclab_rewards.joint_vel_limits                   # Penalty for exceeding velocity limits.
penalize_jvel_norm = spot_rewards.joint_velocity_penalty                   # Norm penalty for joint velocity errors.

# Joint acceleration regularization.
penalize_jacc_l2 = isaaclab_rewards.joint_acc_l2                           # L2 penalty for joint accelerations.
penalize_jacc_norm = spot_rewards.joint_acceleration_penalty               # Norm penalty for joint acceleration errors.

reward_left_right_symmetry = symmetry.rew_left_right_total2zero
reward_hip_knee_symmetry = symmetry.rew_hip_knee_pitch_total2zero
reward_equals_symmetry = symmetry.reward_equals_symmetry
reward_hip_roll_symmetry = symmetry.rew_hip_roll_total2zero # no zero
reward_pose_mean_var_symmetry = symmetry.PoseMeanVariance
reward_pose_mean_min_var_max = symmetry.MeanMinVarianceMax

reward_episode = statistics.PositionStatistics
reward_episode2zero = statistics.Episode2Zero
reward_step = statistics.StepPositionStats

rew_pitch_total2zero = statisticsv2_pos.rew_pitch_total2zero  # Reward for pitch angle deviation from zero.
rew_step_mean_mean = statisticsv2_pos.rew_step_mean_mean  # Reward for mean step position.
rew_step_variance_mean = statisticsv2_pos.rew_step_variance_mean  # Reward for mean step variance.
rew_step_vv_mv = statisticsv2_pos.rew_step_vv_mv  # Reward for mean step velocity variance.
rew_episode_mean = statisticsv2_pos.rew_episode_mean  # Reward for mean episode position.
rew_episode_mean_symmetry = statisticsv2_pos.rew_episode_mean_symmetry  # Reward for symmetry in episode mean position.
rew_episode_variance = statisticsv2_pos.rew_episode_variance  # Reward for episode variance.
rew_episode_variance_symmetry = statisticsv2_pos.rew_episode_variance_symmetry  # Reward for symmetry in episode variance.

rew_mean_self = statisticsv3_pos.rew_mean_self
rew_mean_zero = statisticsv3_pos.rew_mean_zero
rew_mean_zero_nostep = statisticsv3_pos.rew_mean_zero_nostep
rew_mean_zero_nosymmetry = statisticsv3_pos.rew_mean_zero_nosymmetry
rew_variance_self = statisticsv3_pos.rew_variance_self
rew_variance_self_noencourage = statisticsv3_pos.rew_variance_self_noencourage
rew_variance_zero = statisticsv3_pos.rew_variance_zero
rew_variance_zero_nosymmetry = statisticsv3_pos.rew_variance_zero_nosymmetry


"""
Action Penalties:
Penalties to enforce smoothness and regularity in the agent's actions.
"""
penalize_action_rate_l2 = isaaclab_rewards.action_rate_l2                  # L2 penalty on the rate of change of actions.
penalize_action_rate_norm = spot_rewards.action_smoothness_penalty         # Norm penalty to smooth action changes.
penalize_action_smoothness = actions.penalize_action_smoothness            # Direct smoothness penalty for actions.
penalize_action_l2 = isaaclab_rewards.action_l2                            # L2 penalty on action magnitude.

'''
Feet Rewards and Penalties:
These parameters handle rewards and penalties related to feet dynamics, such as air time, sliding, and clearance.
'''
# Air time rewards/penalties.
reward_air_time = loc_rewards.feet_air_time                               # Reward for being airborne.
reward_air_time_phase = feet_phase.reward_feet_air_time                    # Phase-based air time reward.
reward_air_time_biped = loc_rewards.feet_air_time_positive_biped           # Reward for biped air time.
reward_air_time2 = spot_rewards.air_time_reward                            # Alternative air time reward.
penalize_airborne = feet_contact.penalty_feet_airborne                      # Penalty for feet being airborne too long.
penalize_air_time_variance = spot_rewards.air_time_variance_penalty           # Penalty based on variability in air time.

# Slide penalties.
penalize_slide = loc_rewards.feet_slide                                    # Penalty for feet sliding.
penalize_slide_threshold = feet.penalize_feet_slide                         # Penalty applied at sliding threshold.
penalize_slide_threshold2 = spot_rewards.foot_slip_penalty                   # Additional slip penalty.
penalty_stumble = feet_contact.penalty_feet_stumble                         # Penalty for stumbling.
reward_clearance = spot_rewards.foot_clearance_reward                      # Reward for adequate foot clearance.
reward_clearance_phase = feet_phase.reward_feet_clearance                   # Phase-based clearance reward.
penalize_clearance = feet.penalize_foot_clearance                           # Penalty for insufficient clearance.

penalize_statistics_clearance = feet_statisticsv3.penalize_footclearance
# Gait reward.
GaitReward = spot_rewards.GaitReward                                       # Reward for achieving desired gait pattern.

"""
Contact Sensor and Foot Forces:
Penalties/rewards for handling contacts and excessive forces on the feet.
"""
penalize_undesired_contacts = isaaclab_rewards.undesired_contacts           # Penalty for undesired contacts.
penalize_contacts = contact.penalize_collision                            # Collision penalty.
reward_forces_z = feet_contact.reward_feet_forces_z                        # Reward based on vertical forces.
penalize_forces = isaaclab_rewards.contact_forces                           # Penalty for excessive contact forces.
penalize_forces2 = feet_contact.penalize_feet_forces                        # Additional penalty for contact forces.
reward_contact_with_phase = feet_phase.rew_contact_with_phase                # Reward for synchronizing contacts with phase.
reward_contact_with_phase_number = feet_phase.reward_feet_contact_number     # Reward based on the count of contact events.

reward_forces = feet_statistics.ContactStatistics

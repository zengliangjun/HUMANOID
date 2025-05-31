from isaaclab.envs.mdp import rewards as isaaclab_rewards
from isaaclab_tasks.manager_based.classic.cartpole.mdp import rewards as cartpole_rewards
from isaaclab_tasks.manager_based.classic.humanoid.mdp import rewards as humanoid_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import rewards as loc_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp import rewards as spot_rewards

from . import actions, base, base_phase, body, joint, feet_contact, feet, feet_phase, joint_with_phase
'''
episodic
'''
is_alive = isaaclab_rewards.is_alive
is_terminated = isaaclab_rewards.is_terminated
is_terminated_term = isaaclab_rewards.is_terminated_term

"""
base penalties.
"""
flat_orientation_l2 = isaaclab_rewards.flat_orientation_l2
upright_posture_bonus = humanoid_rewards.upright_posture_bonus
base_orientation_penalty = spot_rewards.base_orientation_penalty
reward_orientation = base.reward_orientation

base_height_l2 = isaaclab_rewards.base_height_l2
penalize_base_height = base.penalize_base_height
reward_base_height = base_phase.reward_base_height

lin_vel_z_l2 = isaaclab_rewards.lin_vel_z_l2
ang_vel_xy_l2 = isaaclab_rewards.ang_vel_xy_l2
base_motion_penalty = spot_rewards.base_motion_penalty

body_lin_acc_l2 = isaaclab_rewards.body_lin_acc_l2
reward_body_acc = base.reward_body_acc

move_to_target_bonus = humanoid_rewards.move_to_target_bonus
progress_reward = humanoid_rewards.progress_reward

reward_distance = body.reward_distance

"""
Joint penalties.
"""
joint_torques_l2 = isaaclab_rewards.joint_torques_l2
power_consumption = humanoid_rewards.power_consumption
joint_torques_penalty = spot_rewards.joint_torques_penalty

energy_cost = joint.energy_cost

joint_vel_l1 = isaaclab_rewards.joint_vel_l1
joint_vel_l2 = isaaclab_rewards.joint_vel_l2
joint_vel_limits = isaaclab_rewards.joint_vel_limits
joint_velocity_penalty = spot_rewards.joint_velocity_penalty

joint_pos_limits = isaaclab_rewards.joint_pos_limits
joint_pos_limits_penalty_ratio = humanoid_rewards.joint_pos_limits_penalty_ratio

joint_pos_target_l2 = cartpole_rewards.joint_pos_target_l2

joint_deviation_l1 = isaaclab_rewards.joint_deviation_l1
reward_yaw_rool_joint_pos = joint.reward_yaw_rool_joint_pos

joint_acc_l2 = isaaclab_rewards.joint_acc_l2
joint_acceleration_penalty = spot_rewards.joint_acceleration_penalty
joint_position_penalty = spot_rewards.joint_position_penalty

rew_joint_pos_withrefpose = joint_with_phase.rew_joint_pos_withrefpose
'''
feet
'''
feet_air_time = loc_rewards.feet_air_time
feet_air_time_positive_biped = loc_rewards.feet_air_time_positive_biped
air_time_reward = spot_rewards.air_time_reward
reward_feet_air_time = feet_phase.reward_feet_air_time

penalty_feet_airborne = feet_contact.penalty_feet_airborne

air_time_variance_penalty = spot_rewards.air_time_variance_penalty

feet_slide = loc_rewards.feet_slide
penalize_feet_slide = feet.penalize_feet_slide
foot_slip_penalty = spot_rewards.foot_slip_penalty
penalty_feet_stumble = feet_contact.penalty_feet_stumble

foot_clearance_reward = spot_rewards.foot_clearance_reward
penalize_foot_clearance = feet.penalize_foot_clearance
reward_feet_clearance = feet_phase.reward_feet_clearance

reward_feet_forces_z = feet_contact.reward_feet_forces_z
penalize_feet_forces = feet_contact.penalize_feet_forces

GaitReward = spot_rewards.GaitReward
"""
Contact sensor.
"""
undesired_contacts = isaaclab_rewards.undesired_contacts
penalize_collision = body.penalize_collision

contact_forces = isaaclab_rewards.contact_forces

rew_contact_with_phase = feet_phase.rew_contact_with_phase
reward_feet_contact_number = feet_phase.reward_feet_contact_number
"""
Action penalties.
"""
applied_torque_limits = isaaclab_rewards.applied_torque_limits
action_rate_l2 = isaaclab_rewards.action_rate_l2
action_smoothness_penalty = spot_rewards.action_smoothness_penalty
penalize_action_smoothness = actions.penalize_action_smoothness

action_l2 = isaaclab_rewards.action_l2

"""
Velocity-tracking rewards.
"""
track_lin_vel_xy_exp = isaaclab_rewards.track_lin_vel_xy_exp
track_ang_vel_z_exp = isaaclab_rewards.track_ang_vel_z_exp

track_lin_vel_xy_yaw_frame_exp = loc_rewards.track_lin_vel_xy_yaw_frame_exp
track_ang_vel_z_world_exp = loc_rewards.track_ang_vel_z_world_exp

base_angular_velocity_reward = spot_rewards.base_angular_velocity_reward
base_linear_velocity_reward = spot_rewards.base_linear_velocity_reward

reward_mismatch_vel_exp = base.reward_mismatch_vel_exp
reward_mismatch_speed = base.reward_mismatch_speed
reward_track_vel_hard = base.reward_track_vel_hard

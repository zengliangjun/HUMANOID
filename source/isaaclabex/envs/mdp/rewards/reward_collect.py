from isaaclab.envs.mdp import rewards as isaaclab_rewards
from isaaclab_tasks.manager_based.classic.cartpole.mdp import rewards as cartpole_rewards
from isaaclab_tasks.manager_based.classic.humanoid.mdp import rewards as humanoid_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import rewards as loc_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp import rewards as spot_rewards

from . import actions, base, base_phase, body, joint, feet_contact, feet, feet_phase, joint_with_phase

from . import base_ori
'''
episodic
'''
rewards_eps_alive = isaaclab_rewards.is_alive
penalize_eps_terminated = isaaclab_rewards.is_terminated
penalize_eps_terminated_term = isaaclab_rewards.is_terminated_term

"""
base
"""
# orientation
penalize_ori_l2 = isaaclab_rewards.flat_orientation_l2       # projected_gravity_b  square  sum
penalize_ori_norm = spot_rewards.base_orientation_penalty    # projected_gravity_b  norm
reward_ori_euler_gravity_b = base_ori.reward_orientation_euler_gravity_b
reward_ori_gravity = humanoid_rewards.upright_posture_bonus  # calcute gravity_b z > threshold

# height
penalize_height_flat_or_rayl2 = isaaclab_rewards.base_height_l2         # for flat & ray screan
penalize_height_base2feet = base.penalize_base_height                   # relation base feet height
reward_height_base2feet_phase = base_phase.reward_base_height           # relation base stance feet height,

# mismatch velocity tracking
penalize_lin_z_l2 = isaaclab_rewards.lin_vel_z_l2                      # for vel z
penalize_ang_xy_l2 = isaaclab_rewards.ang_vel_xy_l2                    # for ang xy
penalize_motion_lin_ang = spot_rewards.base_motion_penalty             # for vel z & ang xy

reward_motion_lin_ang = base.reward_mismatch_vel_exp                   # for vel z & ang xy
reward_motion_speed = base.reward_mismatch_speed                       # for vel xy & ang z
reward_motion_hard = base.reward_track_vel_hard                        # for vel xy & ang z hard match

# velocity-tracking rewards.
reward_lin_xy_exp = isaaclab_rewards.track_lin_vel_xy_exp              # for vel xy & ang z, std is sqrt
reward_ang_z_exp = isaaclab_rewards.track_ang_vel_z_exp

reward_lin_xy_exp2 = loc_rewards.track_lin_vel_xy_yaw_frame_exp        # sample as reward_lin_xy_exp
reward_ang_z_exp2 = loc_rewards.track_ang_vel_z_world_exp

reward_lin_xy_exp3 = spot_rewards.base_linear_velocity_reward          # sample as reward_lin_xy_exp, std is ident
reward_ang_z_exp3 = spot_rewards.base_angular_velocity_reward

# target pos
reward_move_to_target_bonus = humanoid_rewards.move_to_target_bonus
reward_progress_reward = humanoid_rewards.progress_reward
# acc regularization
reward_base_acc_exp_norm = base.reward_base_acc


"""
body
"""
# acc regularization
penalize_body_lin_acc_l2 = isaaclab_rewards.body_lin_acc_l2
reward_body_distance = body.reward_distance

"""
Joint penalties.
"""
# energy
penalize_torques_l2 = isaaclab_rewards.joint_torques_l2
penalize_torques_norm = spot_rewards.joint_torques_penalty
penalize_action_vel = humanoid_rewards.power_consumption     # action * vel
penalize_energy = joint.energy_cost
penalize_torque_limits = isaaclab_rewards.applied_torque_limits
# pos
penalize_jpos_limits_l1 = isaaclab_rewards.joint_pos_limits
try:
    penalize_jpos_limits_ratio = humanoid_rewards.joint_pos_limits_penalty_ratio
except:
    penalize_jpos_limits_ratio = humanoid_rewards.joint_limits_penalty_ratio
# pos regularization
penalize_jpos_deviation_l1 = isaaclab_rewards.joint_deviation_l1
penalize_jacc_norm_stand_check = spot_rewards.joint_position_penalty
reward_jpos_yaw_rool = joint.reward_yaw_rool_joint_pos        # reward yaw rool penalize other
reward_jpos_withrefpose = joint_with_phase.rew_joint_pos_withrefpose

# pos target
penalize_jpos_target_l2 = cartpole_rewards.joint_pos_target_l2

# vel regularization
penalize_jvel_l1 = isaaclab_rewards.joint_vel_l1
penalize_jvel_l2 = isaaclab_rewards.joint_vel_l2
penalize_jvel_limits = isaaclab_rewards.joint_vel_limits
penalize_jvel_norm = spot_rewards.joint_velocity_penalty

# acc regularization
penalize_jacc_l2 = isaaclab_rewards.joint_acc_l2
penalize_jacc_norm = spot_rewards.joint_acceleration_penalty
"""
Action penalties.
"""
penalize_action_rate_l2 = isaaclab_rewards.action_rate_l2
penalize_action_rate_norm = spot_rewards.action_smoothness_penalty
penalize_action_smoothness = actions.penalize_action_smoothness # smoothness of action
# action regularization
penalize_action_l2 = isaaclab_rewards.action_l2


'''
feet
'''
# air time
reward_air_time = loc_rewards.feet_air_time
reward_air_time_phase = feet_phase.reward_feet_air_time
reward_air_time_biped = loc_rewards.feet_air_time_positive_biped
reward_air_time2 = spot_rewards.air_time_reward

penalize_airborne = feet_contact.penalty_feet_airborne
penalize_air_time_variance = spot_rewards.air_time_variance_penalty  # self supervision, sample time
# slide
penalize_slide = loc_rewards.feet_slide
penalize_slide_threshold = feet.penalize_feet_slide  # clear
penalize_slide_threshold2 = spot_rewards.foot_slip_penalty

penalty_stumble = feet_contact.penalty_feet_stumble

reward_clearance = spot_rewards.foot_clearance_reward
reward_clearance_phase = feet_phase.reward_feet_clearance

penalize_clearance = feet.penalize_foot_clearance


# gait
GaitReward = spot_rewards.GaitReward
"""
Contact sensor.
"""
penalize_undesired_contacts = isaaclab_rewards.undesired_contacts  # threshold
penalize_contacts = body.penalize_collision

# feet
reward_forces_z = feet_contact.reward_feet_forces_z
penalize_forces = isaaclab_rewards.contact_forces
penalize_forces2 = feet_contact.penalize_feet_forces

reward_contact_with_phase = feet_phase.rew_contact_with_phase
reward_contact_with_phase_number = feet_phase.reward_feet_contact_number

from .joint import joint_pbrs
from .feet import feet_pbrs
from .root import penalize_pbrs, rewards_pbrs

# Feet PBRs
slide_pbrs = feet_pbrs.slide_pbrs
clearance_pbrs = feet_pbrs.clearance_pbrs

# Joint PBRs
jacc_l2_pbrs = joint_pbrs.jacc_l2_pbrs
jvel_l2_pbrs = joint_pbrs.jvel_l2_pbrs
jpos_limits_l1_pbrs = joint_pbrs.jpos_limits_l1_pbrs
jpos_deviation_l1_pbrs = joint_pbrs.jpos_deviation_l1_pbrs
torques_l2_pbrs = joint_pbrs.torques_l2_pbrs

total2zero_pbrs = joint_pbrs.total2zero_pbrs
equals_pbrs = joint_pbrs.equals_pbrs
# Base PBRs
lin_z_pbrs = penalize_pbrs.lin_z_pbrs
ang_xy_pbrs = penalize_pbrs.ang_xy_pbrs
ori_l2_pbrs = penalize_pbrs.ori_l2_pbrs
height_flat_or_rayl2_pbrs = penalize_pbrs.height_flat_or_rayl2_pbrs

lin_xy_exp_pbrs = rewards_pbrs.lin_xy_exp_pbrs
ang_z_exp_pbrs = rewards_pbrs.ang_z_exp_pbrs

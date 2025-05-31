from . import privileged_joins, privileged_feet, privileged_body, privileged_phase

joint_acc = privileged_joins.joint_acc
joint_stiffness = privileged_joins.joint_stiffness
joint_damping = privileged_joins.joint_damping
joint_torques = privileged_joins.joint_torques

feet_contact_status = privileged_feet.feet_contact_status
feet_contact_forces = privileged_feet.feet_contact_forces
feet_pos = privileged_feet.feet_pos

body_mass = privileged_body.body_mass
push_force = privileged_body.push_force
push_torque = privileged_body.push_torque
frictions = privileged_body.frictions

stance_status = privileged_phase.stance_status

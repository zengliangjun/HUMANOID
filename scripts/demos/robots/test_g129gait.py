# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate bipedal robots.

.. code-block:: bash

    # Usage

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch

from isaaclab.app import AppLauncher
import os.path as osp
import sys

_root = osp.join(osp.dirname(__file__), "../../../source/")
sys.path.append(_root)

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate bipedal robots.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##

from isaaclabex.assets.robots.unitree_g129 import UNITREE_GO129_CFG  # isort:skip

def _calcute_limits(_limits, _scales):
    _updates = (_limits[:, :, 1] - _limits[:, :, 0]) * (1 - _scales[None, :]) / 2
    _limits[:, :, 0] = _limits[:, :, 0] + _updates
    _limits[:, :, 1] = _limits[:, :, 1] - _updates
    return _limits

def update_robots(robot: Articulation, _phases: torch.Tensor, sim_dt: float):
    _scales = torch.tensor([
        0, 0,  # 0 ## hip_pitch_joint
        0, 0.25,  # 2
        0.25, 0,  # 4
        0, 0,  # 6
        0, 0,  # 8
        0,     # 10
        0, 0,  # 11 ## knee_joint
        0, 0,  # 13
        0, 0,  # 15
        0, 0,  # 17
        0, 0,  # 19
        0, 0,  # 21
        0, 0,  # 23
        0, 0,  # 25
        0, 0,  # 27
        ], device=_phases.device)

    _update_flag = (_scales[None, :] > 0).float()
    _limits = _calcute_limits(robot.data.joint_pos_limits.clone(), _scales) * _update_flag[:, :, None]
    _limits_range = _limits[:, :, 1] - _limits[:, :, 0]

    _update_pos = _limits_range * torch.sin(_phases[None, :] * 2 * torch.pi) / 2 #+ _limits[:, :, 0]
    print(_update_pos[0, [3, 4]])

    _default_pose = robot.data.default_joint_pos #(1 - _update_flag) * robot.data.default_joint_pos
    #print(_update_pos)

    _joint_pos = _default_pose # _update_pos + _default_pose
    robot.set_joint_position_target(_joint_pos)
    robot.write_data_to_sim()


def design_scene(sim: sim_utils.SimulationContext) -> tuple[list, torch.Tensor]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Define origins
    origins = torch.tensor([
        [0.0, 0.0, 0.0],
    ]).to(device=sim.device)

    # Robots
    UNITREE_GO129_CFG.init_state.pos = (0.0, 0.0, 1)
    UNITREE_GO129_CFG.spawn.articulation_props.fix_root_link = True
    g1 = Articulation(UNITREE_GO129_CFG.replace(prim_path="/World/G1"))
    robots = [g1]

    return robots, origins


def run_simulator(sim: sim_utils.SimulationContext, robots: list[Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # print(robots[0].data.joint_names)
    #print(robots[0].data.default_joint_pos) # b * n
    #print(robots[0].data.joint_pos_limits) # b * n * 2
    for _id in range(len(robots[0].data.joint_names)):
        print(_id, robots[0].data.joint_names[_id],
              robots[0].data.default_joint_pos[0, _id].item(),
              robots[0].data.joint_pos_limits[0, _id, 0].item(),
              robots[0].data.joint_pos_limits[0, _id, 1].item())

    _phases = torch.tensor([
        0.5, 0,  # 0 ## hip_pitch_joint
        0, 0,  # 2
        0, 0,  # 4
        0, 0,  # 6
        0, 0,  # 8
        0,     # 10
        0, 0.5,  # 11 ## knee_joint
        0, 0,  # 13
        0, 0,  # 15
        0, 0,  # 17
        0, 0,  # 19
        0, 0,  # 21
        0, 0,  # 23
        0, 0,  # 25
        0, 0,  # 27
        ], device=sim.device)


    # Simulate physics
    while simulation_app.is_running():
        _step_phase = sim_dt * 1 # frequencie
        _phases = _phases + _step_phase

        torch.clip(_phases, min=0, out=_phases)
        torch.remainder(_phases, 1, out=_phases)

        update_robots(robots[0], _phases, sim_dt)
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in robots:
            robot.update(sim_dt)

        #print(robots[0].data.applied_torque)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.0, 0.0, 2.25], target=[0.0, 0.0, 1.0])

    # design scene
    robots, origins = design_scene(sim)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run the simulator
    run_simulator(sim, robots, origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

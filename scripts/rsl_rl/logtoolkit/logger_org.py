# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
import torch
class Logger:
    def __init__(self, env, comand = "base_velocity", robot_id = 0, joint_id = 0):
        self.env = env
        self.command = comand
        self.robot_id = robot_id
        self.joint_id = joint_id
        self.asset = env.scene["robot"]

        dt = self.env.step_dt

        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_step(self, actions, extra):
        commands = self.env.command_manager.get_command(self.command)

        self._log_states(
            {
                'dof_pos_target': actions[self.robot_id, self.joint_id].item() * self.env.cfg.actions.joint_pos.scale,
                'dof_pos': self.asset.data.joint_pos[self.robot_id, self.joint_id].item(),
                'dof_vel': self.asset.data.joint_vel[self.robot_id, self.joint_id].item(),
                'dof_torque': self.asset.data.applied_torque[self.robot_id, self.joint_id].item(),
                'command_x': commands[self.robot_id, 0].item(),
                'command_y': commands[self.robot_id, 1].item(),
                'command_yaw': commands[self.robot_id, 2].item(),
                'base_vel_x': self.asset.data.root_lin_vel_b[self.robot_id, 0].item(),
                'base_vel_y': self.asset.data.root_lin_vel_b[self.robot_id, 1].item(),
                'base_vel_z': self.asset.data.root_lin_vel_b[self.robot_id, 2].item(),
                'base_vel_yaw': self.asset.data.root_ang_vel_b[self.robot_id, 2].item()
            }
        )

        num_episodes = torch.sum(self.env.reset_buf).item()
        if num_episodes>0:

            if "episode" in extra:
                ep_infos = extra["episode"]
            elif "log" in extra:
                ep_infos = extra["log"]

            #self._log_rewards(ep_infos, num_episodes)
            #self._print_rewards()
            self._plot_states()
            self._reset()


    def _log_state(self, key, value):
        self.state_log[key].append(value)

    def _log_states(self, dict):
        for key, value in dict.items():
            self._log_state(key, value)

    def _log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def _reset(self):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.num_episodes = 0

    def _plot_states(self):
        self.plot_process = Process(target=self._plot, args=([self.state_log]))
        self.plot_process.start()

    def _plot(self, state):

        nb_rows = 9
        nb_cols = 1
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in state.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break

        log = state

        # plot base vel x
        a = axs[0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()
        # plot base vel y
        a = axs[1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()
        # plot base vel yaw
        a = axs[2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()

        # plot joint targets and measured positions
        a = axs[3]
        if log["dof_pos"]: a.plot(time, log["dof_pos"], label='measured')
        if log["dof_pos_target"]: a.plot(time, log["dof_pos_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend()
        # plot joint velocity
        a = axs[4]
        if log["dof_vel"]: a.plot(time, log["dof_vel"], label='measured')
        if log["dof_vel_target"]: a.plot(time, log["dof_vel_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        # plot base vel z
        a = axs[5]
        if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='measured')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        a.legend()
        # plot contact forces
        a = axs[6]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()
        # plot torque/vel curves
        a = axs[7]
        if log["dof_vel"]!=[] and log["dof_torque"]!=[]: a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        a.legend()
        # plot torques
        a = axs[8]
        if log["dof_torque"]!=[]: a.plot(time, log["dof_torque"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()
        plt.show()

    def _print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
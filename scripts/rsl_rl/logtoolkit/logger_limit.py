from .logger_base import LoggerBase
from isaaclab.managers import SceneEntityCfg
import torch
import matplotlib.pyplot as plt
import numpy as np

class LoggerLimits(LoggerBase):
    def __init__(self, env):
        super().__init__(env)
        self.command = "base_velocity"
        self.asset_cfg: SceneEntityCfg = SceneEntityCfg("robot",
                                            joint_names=[
                                                    "left_hip_pitch_joint",
                                                    "right_hip_pitch_joint",
                                                    "left_hip_roll_joint",
                                                    "right_hip_roll_joint",
                                                    "left_hip_yaw_joint",
                                                    "right_hip_yaw_joint",
                                                    "left_knee_joint",
                                                    "right_knee_joint",
                                                    "left_ankle_pitch_joint",
                                                    "right_ankle_pitch_joint",
                                                    "left_ankle_roll_joint",
                                                    "right_ankle_roll_joint",

                                            ],
                                            preserve_order = True)
        self.asset_cfg.resolve(env.scene)
        if slice(None) == self.asset_cfg.joint_ids:
            self.asset_cfg.joint_ids, _ = self.asset.find_joints(self.asset_cfg.joint_names, preserve_order=self.asset_cfg.preserve_order)

    def _calcute_states(self, actions, extra):
        states = {}
        commands = self.env.command_manager.get_command(self.command)

        states['cxy'] = torch.norm(commands[self.robot_id, :2]).item()
        states['vxy'] = torch.norm(self.asset.data.root_lin_vel_b[self.robot_id, :2]).item()
        states['cyaw'] = commands[self.robot_id, 2].item()
        states['vyaw'] = self.asset.data.root_lin_vel_b[self.robot_id, 2].item()

        for id, joint_name in enumerate(self.asset_cfg.joint_names):
            pos = self.asset.data.joint_pos[self.robot_id, self.asset_cfg.joint_ids[id]].item()
            difmin = min(pos -  self.asset.data.soft_joint_pos_limits[self.robot_id, self.asset_cfg.joint_ids[id], 0].item(), 0)
            difmax = max(pos -  self.asset.data.soft_joint_pos_limits[self.robot_id, self.asset_cfg.joint_ids[id], 1].item(), 0)

            states[f'{joint_name}'] = difmin + difmax
        return states

    def _plot(self, state):
        keys = list(state.keys())
        nb_rows = len(keys) // 2
        nb_cols = 1
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in state.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break

        for id in range(nb_rows):
            a = axs[id]
            key = keys[id * 2]
            a.plot(time, state[key], label=key)
            key = keys[id * 2 + 1]
            a.plot(time, state[key], label=key)
            a.set(xlabel='time [s]', ylabel='[rad]')
            a.legend()

        plt.show()

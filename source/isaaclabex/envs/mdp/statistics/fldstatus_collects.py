from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from rsl_rl.modules import EmpiricalNormalization
from rsl_rlex.fld.modules import modules_cfg, fld_models

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclabex.envs.managers.term_cfg import StatisticsTermCfg

"""

@configclass
class FLDCfg(modules_cfg.FLDCfg):
    step_dt = 0.02

    observation_dim = 27
    observation_history_horizon = 51

    encoder_hidden_dims = [64, 64, 8]
    decoder_hidden_dims = [8, 64, 64]

cfg = FLDCfg()

fld_status = term_cfg.StatisticsTermCfg(
        func= FLDCollect,
        params={
            "training": False,
            # "log_dir": "",   # env.log_dir
            "training_noise_level": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
            "status_names": [
                "ang_vel",      # 0.5
                "gravity",      # 1
                "joint_pos",    # 1
                "joint_vel",    # 0.5
            ],
            "fld_module_cfg": cfg,
            "fld_loss_scales": [],
            "fld_learning_rate": 0.0001,
            "fld_weight_decay": 0.0005,
        },
    )

"""

class FldNormalization(EmpiricalNormalization):

    def __init__(self, shape, eps=1e-2, until=None):
        super(FldNormalization, self).__init__(shape, eps = eps, until = until)

    def forward(self, inputs: torch.Tensor):

        if self.training:
            sizes = inputs.shape
            inputs_reshape = torch.reshape(inputs, (-1, sizes[-1]))
            valids = torch.sum(inputs_reshape, dim = -1) > 0
            count = torch.sum(valids.float())
            if count > 0:
                valids = inputs_reshape[valids]
                self.update(valids)

        return (inputs - self._mean) / (self._std + self.eps)


class FLDCollect(ManagerTermBase):

    cfg: StatisticsTermCfg

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        params: dict = cfg.params

        asset_cfg: SceneEntityCfg = params["asset_cfg"]
        self.asset: Articulation = self._env.scene[asset_cfg.name]
        assert "fld_module_cfg" in params
        fld_cfg : modules_cfg.FLDCfg = params["fld_module_cfg"]
        fld_cfg.step_dt = env.step_dt

        observation_dim = fld_cfg.observation_dim
        observation_history_horizon = fld_cfg.observation_history_horizon

        self.history_status = torch.zeros((self.num_envs, \
                                           observation_history_horizon, \
                                           observation_dim), \
                                           device = self.device, \
                                           dtype = torch.float32)

        self.fld_params = torch.zeros((self.num_envs, \
                                           fld_cfg.encoder_hidden_dims[-1], \
                                           5), \
                                           device = self.device, \
                                           dtype = torch.float32)

        ## module
        self.fld_module = fld_models.FLD(fld_cfg).to(env.device)
        self.status_normalizer = FldNormalization(shape = [observation_dim], until = 1.0e8).to(env.device)


        self.training = params["training"]
        if self.training:
            self.fld_module.train()
            self.status_normalizer.train()

            self.fld_optimizer = optim.Adam(self.fld_module.parameters(), \
                                            lr = params["fld_learning_rate"], \
                                            weight_decay = params["fld_weight_decay"])

            assert hasattr(env.cfg, "log_dir")
            self.writer = SummaryWriter(log_dir = env.cfg.log_dir, flush_secs = 10)

            self.training_noise_level = params["training_noise_level"]
            self.fld_loss_scales = torch.tensor(params["fld_loss_scales"], dtype = torch.float32, device = env.device)[None, :]

        else:
            self.fld_module.eval()
            self.status_normalizer.eval()



    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        """重置指定环境的统计缓冲区
        Args:
            env_ids: 需要重置的环境ID列表
        Returns:
            空字典（保持接口统一）
        """
        if env_ids is None or len(env_ids) == 0:
            return {}

        self.history_status[env_ids, ...] = 0

        return {}

    def _get_ang_vel(self):
        return self.asset.data.root_ang_vel_b

    def _get_gravity(self):
        return self.asset.data.projected_gravity_b

    def _get_joint_pos(self):
        return self.asset.data.joint_pos - self.asset.data.default_joint_pos

    def _get_joint_vel(self):
        return self.asset.data.joint_vel - self.asset.data.default_joint_vel

    # TRAIN
    def _compute_loss(self, input, target):
        diff = torch.square((input - target) * self.fld_loss_scales)
        return torch.mean(torch.sum(diff, dim=-1))

    def _step_training(self, pre_status, cur_status):
        inputs = self.status_normalizer(pre_status)

        inputs_noised = inputs + torch.randn_like(inputs, device = self.device) * self.training_noise_level
        forecast_dynamics, latent, signal, params = self.fld_module.forward(inputs_noised, forecast_horizon = 2)
        forecast_status = self.status_normalizer.inverse(forecast_dynamics)

        loss = self._compute_loss(forecast_status[0, ...], pre_status)
        loss += self._compute_loss(forecast_status[1, ...], cur_status)

        self.fld_optimizer.zero_grad()
        loss.backward()
        self.fld_optimizer.step()


    def __call__(self):
        """执行统计计算"""
        status_list = []
        for status in self.cfg.params["status_names"]:
            status_fun = getattr(self, f"_get_{status}")
            status_list.append(status_fun())
        status_list = torch.cat(status_list, dim = -1)

        if self.training:
            pre_history_status = self.history_status.clone()

        ## update fld params
        self.history_status[:, :-1] = self.history_status[:, 1:].clone()
        self.history_status[:, -1] = status_list

        if self.training:
            self._step_training(pre_history_status, self.history_status)

        # with torch.no_grad():
        with torch.inference_mode():
            status = self.status_normalizer(self.history_status)
            latent, params = self.fld_module.forward_encod(status)

            phase, frequency, amplitude, offset = params
            phase += frequency * self._env.step_dt

            self.fld_params[...] = torch.cat(
                        (torch.sin(2.0 * torch.pi * phase)[:, :, None],
                        torch.cos(2.0 * torch.pi * phase)[:, :, None],
                        frequency[:, :, None],
                        amplitude[:, :, None],
                        offset[:, :, None]
                        ), dim = -1)

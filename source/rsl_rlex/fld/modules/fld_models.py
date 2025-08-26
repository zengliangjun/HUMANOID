## copy from https://github.com/mit-biomimetics/fld.git
"""
Fourier Latent Dynamics (FLD) model implementation.
This module implements encoder-decoder architecture with Fourier analysis for
learning latent dynamics from time-series observations.
"""

from torch import nn
import torch
import copy

try:
    from rsl_rlex.fld.modules import modules_cfg
except:
    import os.path as osp
    import sys

    from isaaclab.utils import configclass
    root = osp.dirname(__file__)
    root = osp.join(root, "../../..")
    sys.path.append(root)
    from rsl_rlex.fld.modules import modules_cfg

class FLDEncoder(nn.Module):
    """
    FLD Encoder module that processes time-series observations into latent space.
    Uses 1D convolutions to extract features and Fourier analysis to capture
    frequency domain characteristics.

    Args:
        cfg (modules_cfg.FLDCfg): Configuration containing model hyperparameters
    """
    cfg: modules_cfg.FLDCfg

    def __init__(self, cfg: modules_cfg.FLDCfg):
        """Initialize encoder with convolutional layers and phase encoders"""
        super(FLDEncoder, self).__init__()
        self.cfg = cfg

        encoder_layers = []
        curr_in_channel = cfg.observation_dim
        for hidden_channel in cfg.encoder_hidden_dims:
            encoder_layers.append(
                nn.Conv1d(
                    curr_in_channel,
                    hidden_channel,
                    cfg.observation_history_horizon,
                    stride=1,
                    padding=int((cfg.observation_history_horizon - 1) / 2),
                    dilation=1,
                    groups=1,
                    bias=True,
                    padding_mode='zeros')
                )
            encoder_layers.append(nn.BatchNorm1d(num_features=hidden_channel))
            encoder_layers.append(nn.ELU())
            curr_in_channel = hidden_channel

        self.encoder = nn.Sequential(*encoder_layers)

        self.phase_encoder = nn.ModuleList()
        for _ in range(curr_in_channel):
            phase_encoder_layers = []
            phase_encoder_layers.append(nn.Linear(cfg.observation_history_horizon, 2))
            phase_encoder_layers.append(nn.BatchNorm1d(num_features=2))
            phase_encoder = nn.Sequential(*phase_encoder_layers)
            self.phase_encoder.append(phase_encoder)

        freqs = torch.fft.rfftfreq(cfg.observation_history_horizon)[1:] * cfg.observation_history_horizon
        self.register_buffer('freqs', freqs, persistent=False)


    def _fft(self, inputs):
        """
        对输入时间序列进行快速傅里叶变换(FFT)分析
        计算每个通道的主频、振幅和直流偏移量

        Compute Fast Fourier Transform on input time-series.

        Args:
            inputs: Input tensor of shape (B, C, T)

        Returns:
            Tuple of (frequency, amplitude, offset) where:
            - frequency: Dominant frequency for each channel
            - amplitude: Signal amplitude for each channel
            - offset: DC offset for each channel
        """
        # 计算实数FFT (只使用实数部分)
        rfft = torch.fft.rfft(inputs, dim=2)

        # 计算幅度谱 (去除直流分量)
        magnitude = rfft.abs()
        spectrum = magnitude[:, :, 1:]  # 去掉直流分量(索引0)
        power = torch.square(spectrum)  # 计算功率谱

        # 计算总功率(用于频率加权)
        power2 = torch.sum(power, dim=2)

        # 计算加权平均频率 (主频)
        frequency = torch.sum(self.freqs * power, dim=2) / power2
        # 计算振幅 (考虑FFT对称性和归一化)
        amplitude = 2 * torch.sqrt(power2) / self.cfg.observation_history_horizon
        # 计算直流偏移量 (FFT的实部第一个元素)
        offset = rfft.real[:, :, 0] / self.cfg.observation_history_horizon

        return frequency, amplitude, offset

    def forward(self, inputs):
        """
        Forward pass of encoder.

        Args:
            inputs: Input tensor of shape (B, C, T) where:
                B = batch size
                C = observation_dim (input channels)
                T = observation_history_horizon (time steps)

        Returns:
            Tuple of (latent, params) where:
            - latent: Encoded features (B, hidden_dim, T)
            - params: List of [phase, frequency, amplitude, offset] for each channel
        """
        latent_channel = self.cfg.encoder_hidden_dims[-1]
        latent = self.encoder(inputs)  # (B, encoder_hidden_dims[-1], T)

        frequency, amplitude, offset = self._fft(latent)
        phase = torch.zeros((inputs.shape[0], latent_channel), device=inputs.device, dtype=torch.float)
        for i in range(latent_channel):
            phase_shift = self.phase_encoder[i](latent[:, i, :])
            phase[:, i] = torch.atan2(phase_shift[:, 1], phase_shift[:, 0]) / (2 * torch.pi)

        params = [phase, frequency, amplitude, offset] # (batch_size, latent_channel)

        return latent, params


class FLDDecoder(nn.Module):
    """
    FLD Decoder module that reconstructs time-series from latent parameters.
    Uses inverse Fourier transform and convolutional layers to generate predictions.

    Args:
        cfg (modules_cfg.FLDCfg): Configuration containing model hyperparameters
    """
    cfg: modules_cfg.FLDCfg

    def __init__(self, cfg: modules_cfg.FLDCfg):
        """Initialize decoder with transposed convolutional layers"""
        super(FLDDecoder, self).__init__()
        self.cfg = cfg

        decoder_layers = []
        curr_in_channel = cfg.decoder_hidden_dims[0]
        hidden_dims = copy.deepcopy(cfg.decoder_hidden_dims[1: ])
        hidden_dims.append(cfg.observation_dim)
        for id, hidden_channel in enumerate(hidden_dims):
            decoder_layers.append(
                nn.Conv1d(
                    curr_in_channel,
                    hidden_channel,
                    cfg.observation_history_horizon,
                    stride=1,
                    padding=int((cfg.observation_history_horizon - 1) / 2),
                    dilation=1,
                    groups=1,
                    bias=True,
                    padding_mode='zeros')
                )

            if id < len(hidden_dims) - 1:
                decoder_layers.append(nn.BatchNorm1d(num_features=hidden_channel))
                decoder_layers.append(nn.ELU())
            curr_in_channel = hidden_channel

        self.decoder = nn.Sequential(*decoder_layers)

        history_horizon_offset = torch.linspace(- (cfg.observation_history_horizon - 1) * cfg.step_dt / 2,
                                (cfg.observation_history_horizon - 1) * cfg.step_dt / 2,
                                 cfg.observation_history_horizon, dtype = torch.float32)

        self.register_buffer('history_horizon_offset', history_horizon_offset, persistent=False)

    def forward(self, inputs, forecast_horizon = 1):
        """
        Forward pass of decoder.

        Args:
            inputs: Tuple of (phase, frequency, amplitude, offset) from encoder
            forecast_horizon: Number of future time steps to predict

        Returns:
            Tuple of (pred_dynamics, signal) where:
            - pred_dynamics: Predicted future dynamics (forecast_horizon, B, C, T)
            - signal: Reconstructed input signal (B, hidden_dim, T)
        """
        # (batch_size, latent_channel)
        phase, frequency, amplitude, offset = inputs

        phase_dynamics = phase.unsqueeze(0) + \
                         frequency.unsqueeze(0) * self.cfg.step_dt * \
                         torch.arange(0, forecast_horizon, device = phase.device, dtype = torch.float, requires_grad=False).view(-1, 1, 1)
        # (forecast_horizon, batch_size, latent_channel)

        #                 batch_size, latent_channel, history_horizon
        forecast_phase = (frequency.unsqueeze(-1) * self.history_horizon_offset).unsqueeze(0) + phase_dynamics.unsqueeze(-1)
        # (forecast_horizon, batch_size, latent_channel, history_horizon)

        z = amplitude.unsqueeze(-1).unsqueeze(0) * \
            torch.sin(2 * torch.pi * forecast_phase) + offset.unsqueeze(-1).unsqueeze(0)
        # (forecast_horizon, batch_size, latent_channel, history_horizon)

        signal = z[0]
        pred_dynamics = self.decoder(z.flatten(0, 1))
        pred_dynamics = pred_dynamics.view(forecast_horizon, -1, self.cfg.observation_dim, self.cfg.observation_history_horizon)
        # (forecast_horizon, batch_size, input_channel, history_horizon)

        return pred_dynamics, signal


class FLD(nn.Module):
    """
    Fourier Latent Dynamics model combining encoder and decoder.
    Learns latent representations of time-series data using Fourier analysis.

    Args:
        cfg (modules_cfg.FLDCfg): Configuration containing model hyperparameters
    """
    cfg: modules_cfg.FLDCfg

    def __init__(self, cfg: modules_cfg.FLDCfg):
        """Initialize FLD model with encoder and decoder"""
        super(FLD, self).__init__()
        self.cfg = cfg

        assert self.cfg.encoder_hidden_dims[-1] == self.cfg.decoder_hidden_dims[0], \
            f"encoder_hidden_dims[-1] ({self.cfg.encoder_hidden_dims[-1]}) must be equal to decoder_hidden_dims[0] ({self.cfg.decoder_hidden_dims[0]})"

        #assert self.cfg.observation_dim == self.cfg.decoder_hidden_dims[-1], \
        #    f"observation_dim ({self.cfg.observation_dim}) must be equal to decoder_hidden_dims[-1] ({self.cfg.decoder_hidden_dims[-1]})"

        self.encoder = FLDEncoder(cfg)
        self.decoder = FLDDecoder(cfg)

    def forward(self, inputs, forecast_horizon = 1):
        """
        Forward pass of FLD model.

        Args:
            inputs: Input tensor of shape (B, T, C) where:
                B = batch size
                T = observation_history_horizon (time steps)
                C = observation_dim (input channels)
            forecast_horizon: Number of future time steps to predict

        Returns:
            Tuple of (pred_dynamics, latent, signal, params) where:
            - pred_dynamics: Predicted future dynamics
            - latent: Encoded features
            - signal: Reconstructed input signal
            - params: Fourier parameters [phase, frequency, amplitude, offset]
        """
        inputs = inputs.swapaxes(-2, -1)
        latent, params = self.encoder(inputs)
        pred_dynamics, signal = self.decoder(params, forecast_horizon)
        # (forecast_horizon, B, self.cfg.observation_dim, cfg.observation_history_horizon)
        # signal: (B, self.cfg.encoder_hidden_dims[-1], cfg.observation_history_horizon)

        pred_dynamics = pred_dynamics.swapaxes(-2, -1)

        return pred_dynamics, latent, signal, params

    def forward_encod(self, inputs):
        inputs = inputs.swapaxes(-2, -1)
        return self.encoder(inputs)


if __name__ == "__main__":

    @configclass
    class FLDCfg(modules_cfg.FLDCfg):
        step_dt = 0.02

        observation_dim = 27
        observation_history_horizon = 51

        encoder_hidden_dims = [64, 64, 8]
        decoder_hidden_dims = [8, 64, 64]

    cfg = FLDCfg()
    module = FLD(cfg)

    device = "cuda:0"
    module = module.to(device)

    inputs = torch.randn((12, cfg.observation_history_horizon, cfg.observation_dim), \
                         dtype = torch.float32, device = device)

    outs = module(inputs, forecast_horizon = 50)

    print(outs)


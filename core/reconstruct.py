""" Copyright (C) 2021 Pony Preservation Project

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>."""

import sys
import torch
import yaml
from omegaconf import OmegaConf
import numpy as np
import torch.nn.functional as func
from torch import nn
import math
from taming.models.vqgan import VQModel

sys.path.append("hifi-gan")
from meldataset import MAX_WAV_VALUE, mel_spectrogram


class Reconstruct:
    def __init__(self, device, config_path, checkpoint_path):
        self.device = device
        self.normalize_input = False
        self.interpolate = True
        self.vqgan = None
        self.resolution = 0
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

    def _load_config(self, config_path, display=False):
        config = OmegaConf.load(config_path)
        if display:
            print(yaml.dump(OmegaConf.to_container(config)))
        return config, config.model.params.ddconfig.resolution

    def _load_vqgan(self, config, ckpt_path=None):
        model = torch.load(ckpt_path, map_location="cpu")
        return model.eval()

    def _preprocess_spect(self, input, resolution, is_audio):
        if is_audio:
            wave = input / MAX_WAV_VALUE
            wave = torch.FloatTensor(wave).to(torch.device(self.device))
            spect = (
                mel_spectrogram(
                    wave.unsqueeze(0),
                    2048,  # n_fft
                    resolution,  # num_mels
                    22050,  # sampling_rate
                    256,  # hop_size
                    1024,  # win_size
                    0,  # fmin
                    8000,  # fmax
                )
                .cpu()
                .numpy()
            )
        else:
            spect = input.cpu().numpy()
        spect = (spect + 11.512925) / 6.907755  # log(1e-5) to log(1e1) -> 0 to 2
        if self.normalize_input:
            spect /= 2.0
            spect = (spect / np.max(spect)) * 2
            spect *= 0.935  # Reduce loudness to resemble training data
        spect -= 1  # 0 to 2 -> -1 to 1
        spect = torch.FloatTensor(spect)
        if spect.shape[1] != resolution:
            spect = func.interpolate(
                spect.permute(0, 2, 1), size=resolution, mode="linear"
            ).permute(0, 2, 1)
        return spect.unsqueeze(0)

    def _postprocess_spect(self, x):
        spect = x.cpu().squeeze(0).numpy()
        spect = (spect + 1) * 6.907755 - 11.512925
        return spect

    # From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
    def _get_gaussian_kernel(self, kernel_size=3, sigma=2, channels=3):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.0
        variance = sigma ** 2.0

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
        )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
        )

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter

    def _low_high_pass(self, x, kernel_size=5, sigma=2):
        k = kernel_size // 2
        low = func.pad(
            x,
            (k, k, k, k),
            mode="reflect",
        )
        blur = self._get_gaussian_kernel(kernel_size, sigma, 1)
        low = blur(low)
        high = low * (-0.5)
        high = high + x * 0.5
        return low, high

    def _reconstruct_with_vqgan(self, x, model):
        z, _, [_, _, indices] = model.encode(x)
        xrec = model.decode(
            z
        )  # Note that width gets rounded down to nearest multiple of code resolution

        if x.shape[3] != xrec.shape[3]:
            if x.shape[3] < 32:
                return x  # Too short to reconstruct
            tail = x[:, :, :, -32:]
            ztail, _, [_, _, indices] = model.encode(tail)
            tailrec = model.decode(ztail)
            xrec = torch.cat(
                (xrec, tailrec[:, :, :, -(x.shape[3] - xrec.shape[3]) :]), 3
            )
            assert x.shape[3] == xrec.shape[3]
        return xrec

    def reconstruct(self, input, is_audio=False):
        if self.vqgan is None:
            config, self.resolution = self._load_config(self.config_path)
            self.vqgan = self._load_vqgan(
                config,
                ckpt_path=self.checkpoint_path,
            )
        x = self._preprocess_spect(input, self.resolution, is_audio)
        x_low, x_high = self._low_high_pass(x, 19, 8)
        y = self._reconstruct_with_vqgan(x, self.vqgan)
        y_low, y_high = self._low_high_pass(y, 19, 8)
        mix = x_low + y_high * 2
        z = self._postprocess_spect(mix)
        if self.interpolate:
            target_height = 80
            z = torch.FloatTensor(z)
            z = (
                func.interpolate(z.permute(0, 2, 1), size=target_height, mode="linear")
                .permute(0, 2, 1)
                .numpy()
            )
        return torch.from_numpy(z).to(self.device)

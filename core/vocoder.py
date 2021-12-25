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

import json
import os
import sys
import numpy as np
import resampy
import scipy
import torch

sys.path.append("hifi-gan")
from denoiser import Denoiser
from env import AttrDict
from meldataset import MAX_WAV_VALUE, mel_spectrogram
from models import Generator


class HiFiGAN:
    def __init__(self, model_path, conf_name, device):
        # Load HiFi-GAN
        conf = os.path.join("hifi-gan", conf_name + ".json")
        with open(conf) as f:
            json_config = json.loads(f.read())
        self.h = AttrDict(json_config)
        torch.manual_seed(self.h.seed)
        self.hifigan = Generator(self.h).to(torch.device(device))
        state_dict_g = torch.load(model_path, map_location=torch.device(device))
        self.hifigan.load_state_dict(state_dict_g["generator"])
        self.hifigan.eval()
        self.hifigan.remove_weight_norm()
        self.denoiser = Denoiser(self.hifigan, mode="normal")
        self.device = device

    def vocode(self, spect):
        y_g_hat = self.hifigan(spect.float())
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio_denoised = self.denoiser(audio.view(1, -1), strength=35)[:, 0]
        return audio_denoised.detach().cpu().numpy().reshape(-1).astype(np.int16)

    def superres(self, audio, original_sr):
        # Resampling
        wave = resampy.resample(
            audio,
            original_sr,
            self.h.sampling_rate,
            filter="sinc_window",
            window=scipy.signal.windows.hann,
            num_zeros=8,
        )
        wave_out = wave.astype(np.int16)

        # Super-res
        wave = wave / MAX_WAV_VALUE
        wave = torch.FloatTensor(wave).to(torch.device(self.device))
        new_mel = mel_spectrogram(
            wave.unsqueeze(0),
            self.h.n_fft,
            self.h.num_mels,
            self.h.sampling_rate,
            self.h.hop_size,
            self.h.win_size,
            self.h.fmin,
            self.h.fmax,
        )
        y_g_hat2 = self.hifigan(new_mel)
        audio2 = y_g_hat2.squeeze()
        audio2 = audio2 * MAX_WAV_VALUE
        audio2_denoised = self.denoiser(audio2.view(1, -1), strength=35)[:, 0]

        # High-pass filter, mixing and denormalizing
        audio2_denoised = audio2_denoised.detach().cpu().numpy().reshape(-1)
        b = scipy.signal.firwin(
            101, cutoff=10500, fs=self.h.sampling_rate, pass_zero=False
        )
        y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
        y *= 4.0  # superres strength
        y_out = y.astype(np.int16)
        y_padded = np.zeros(wave_out.shape)
        y_padded[: y_out.shape[0]] = y_out
        sr_mix = wave_out + y_padded
        return sr_mix, self.h.sampling_rate

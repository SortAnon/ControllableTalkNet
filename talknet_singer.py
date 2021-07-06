from collections import OrderedDict
from typing import List

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch import nn
from torch.nn import functional as F
import numpy as np

from nemo.collections.asr.data.audio_to_text import AudioToCharWithDursF0Dataset
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.modules.talknet import (
    GaussianEmbedding,
    MaskedInstanceNorm1d,
    StyleResidual,
)
from nemo.core import Exportable
from nemo.core.classes import ModelPT, PretrainedModelInfo, typecheck
from nemo.core.neural_types import MelSpectrogramType, NeuralType


class TalkNetSingerModel(SpectrogramGenerator, Exportable):
    """TalkNet's mel spectrogram prediction pipeline."""

    @property
    def output_types(self):
        return OrderedDict(
            {"mel-spectrogram": NeuralType(("B", "D", "T"), MelSpectrogramType())}
        )

    def __init__(self, cfg: DictConfig, trainer: "Trainer" = None):
        super().__init__(cfg=cfg, trainer=trainer)
        typecheck.set_typecheck_enabled(enabled=False)

        cfg = self._cfg
        self.vocab = AudioToCharWithDursF0Dataset.make_vocab(
            **cfg.train_ds.dataset.vocab
        )
        self.blanking = cfg.train_ds.dataset.blanking
        self.preprocessor = instantiate(cfg.preprocessor)
        self.embed = GaussianEmbedding(self.vocab, cfg.d_char)
        self.norm_f0 = MaskedInstanceNorm1d(1)
        self.res_f0 = StyleResidual(cfg.d_char, 1, kernel_size=3)
        self.model = instantiate(cfg.model)
        d_out = cfg.model.jasper[-1].filters
        self.proj = nn.Conv1d(d_out, cfg.n_mels, kernel_size=1)

    def forward(self, text, text_len, durs, f0):
        x, x_len = self.embed(text, durs).transpose(1, 2), durs.sum(-1)
        f0_sin = np.nan_to_num(np.sin(9.06472 * np.log(0.0363636 * f0.cpu())))
        f0_cos = np.nan_to_num(np.cos(9.06472 * np.log(0.0363636 * f0.cpu())))
        f0_note = torch.stack(
            (torch.from_numpy(f0_sin), torch.from_numpy(f0_cos)), 1
        ).to(self.device)
        f0, f0_mask = f0.clone(), f0 > 0.0
        f0 = self.norm_f0(f0.unsqueeze(1), f0_mask)
        f0[~f0_mask.unsqueeze(1)] = 0.0
        x = self.res_f0(x, f0)
        x = torch.cat((x, f0_note), 1)
        y, _ = self.model(x, x_len)
        mel = self.proj(y)
        return mel

    @staticmethod
    def _metrics(true_mel, true_mel_len, pred_mel):
        loss = F.mse_loss(pred_mel, true_mel, reduction="none").mean(dim=-2)
        mask = get_mask_from_lengths(true_mel_len)
        loss *= mask.float()
        loss = loss.sum() / mask.sum()
        return loss

    def training_step(self, batch, batch_idx):
        audio, audio_len, text, text_len, durs, f0, f0_mask = batch
        mel, mel_len = self.preprocessor(audio, audio_len)
        pred_mel = self(text=text, text_len=text_len, durs=durs, f0=f0)
        loss = self._metrics(true_mel=mel, true_mel_len=mel_len, pred_mel=pred_mel)
        train_log = {"train_loss": loss}
        return {"loss": loss, "progress_bar": train_log, "log": train_log}

    def validation_step(self, batch, batch_idx):
        audio, audio_len, text, text_len, durs, f0, f0_mask = batch
        mel, mel_len = self.preprocessor(audio, audio_len)
        pred_mel = self(text=text, text_len=text_len, durs=durs, f0=f0)
        loss = self._metrics(true_mel=mel, true_mel_len=mel_len, pred_mel=pred_mel)
        val_log = {"val_loss": loss}
        self.log_dict(
            val_log, prog_bar=False, on_epoch=True, logger=True, sync_dist=True
        )

    @staticmethod
    def _loader(cfg):
        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            **cfg.dataloader_params,
        )

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    def parse(self, text: str, **kwargs) -> torch.Tensor:
        return torch.tensor(self.vocab.encode(text)).long().unsqueeze(0).to(self.device)

    def generate_spectrogram(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        assert hasattr(self, "_durs_model") and hasattr(self, "_pitch_model")

        if self.blanking:
            tokens = [
                AudioToCharWithDursF0Dataset.interleave(
                    x=torch.empty(len(t) + 1, dtype=torch.long, device=t.device).fill_(
                        self.vocab.blank
                    ),
                    y=t,
                )
                for t in tokens
            ]
            tokens = AudioToCharWithDursF0Dataset.merge(
                tokens, value=self.vocab.pad, dtype=torch.long
            )

        text_len = torch.tensor(tokens.shape[-1], dtype=torch.long).unsqueeze(0)
        durs = self._durs_model(tokens, text_len)
        durs = durs.exp() - 1
        durs[durs < 0.0] = 0.0
        durs = durs.round().long()

        # Pitch
        f0_sil, f0_body = self._pitch_model(tokens, text_len, durs)
        sil_mask = f0_sil.sigmoid() > 0.5
        f0 = f0_body * self._pitch_model.f0_std + self._pitch_model.f0_mean
        f0 = (~sil_mask * f0).float()

        # Spect
        mel = self(tokens, text_len, durs, f0)

        return mel

    def forward_for_export(self, tokens: torch.Tensor, text_len: torch.Tensor):
        durs = self._durs_model(tokens, text_len)
        durs = durs.exp() - 1
        durs[durs < 0.0] = 0.0
        durs = durs.round().long()

        # Pitch
        f0_sil, f0_body = self._pitch_model(tokens, text_len, durs)
        sil_mask = f0_sil.sigmoid() > 0.5
        f0 = f0_body * self._pitch_model.f0_std + self._pitch_model.f0_mean
        f0 = (~sil_mask * f0).float()

        # Spect
        x, x_len = self.embed(tokens, durs).transpose(1, 2), durs.sum(-1)
        f0_sin = np.nan_to_num(np.sin(9.06472 * np.log(0.0363636 * f0.cpu())))
        f0_cos = np.nan_to_num(np.cos(9.06472 * np.log(0.0363636 * f0.cpu())))
        f0_note = torch.stack(
            (torch.from_numpy(f0_sin), torch.from_numpy(f0_cos)), 1
        ).to(self.device)
        f0, f0_mask = f0.clone(), f0 > 0.0
        f0 = self.norm_f0(f0.unsqueeze(1), f0_mask)
        f0[~f0_mask.unsqueeze(1)] = 0.0
        x = self.res_f0(x, f0)
        x = torch.cat((x, f0_note), 1)
        y, _ = self.model(x, x_len)
        mel = self.proj(y)

        return mel

    def force_spectrogram(
        self, tokens: torch.Tensor, durs: torch.Tensor, f0: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        if self.blanking:
            tokens = [
                AudioToCharWithDursF0Dataset.interleave(
                    x=torch.empty(len(t) + 1, dtype=torch.long, device=t.device).fill_(
                        self.vocab.blank
                    ),
                    y=t,
                )
                for t in tokens
            ]
            tokens = AudioToCharWithDursF0Dataset.merge(
                tokens, value=self.vocab.pad, dtype=torch.long
            )

        text_len = torch.tensor(tokens.shape[-1], dtype=torch.long).unsqueeze(0)
        durs_len = torch.tensor(durs.shape[-1], dtype=torch.long).unsqueeze(0)
        assert text_len == durs_len

        # Spect
        mel = self(tokens, text_len, durs, f0)
        return mel

    @classmethod
    def list_available_models(cls) -> "List[PretrainedModelInfo]":
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_talknet",
            location=(
                "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_talknet/versions/1.0.0rc1/files"
                "/talknet_spect.nemo"
            ),
            description=(
                "This model is trained on LJSpeech sampled at 22050Hz, and can be used to generate female "
                "English voices with an American accent."
            ),
            class_=cls,  # noqa
            aliases=["TalkNet-22050Hz"],
        )
        list_of_models.append(model)
        return list_of_models

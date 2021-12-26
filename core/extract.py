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
import re
import nemo
import torch
from nemo.collections.asr.data.audio_to_text import AudioToCharWithDursF0Dataset
from nemo.collections.asr.models import EncDecCTCModel
from tqdm import tqdm
from core.nemo_functions import backward_extractor, forward_extractor
import numpy as np
from scipy.io import wavfile
import psola
import crepe
import io
import base64
import tensorflow_hub as hub
import tensorflow as tf
import scipy
import resampy
import torchcrepe


USE_SPICE = False  # Better than CREPE in some cases, but a lot worse with noisy audio


class ExtractDuration:
    def __init__(self, run_path, device):
        self.asr_model = (
            EncDecCTCModel.from_pretrained(model_name="asr_talknet_aligner")
            .cpu()
            .eval()
        )
        self.run_path = run_path
        self.arpadict = self._load_dictionary(
            os.path.join(run_path, "horsewords.clean")
        )
        self.parser = AudioToCharWithDursF0Dataset.make_vocab(
            notation="phonemes",
            punct=True,
            spaces=True,
            stresses=False,
            add_blank_at="last",
        )
        self.device = device

    def _load_dictionary(self, dict_path):
        arpadict = dict()
        with open(dict_path, "r", encoding="utf8") as f:
            for line in f.readlines():
                word = line.split("  ")
                assert len(word) == 2
                arpadict[word[0].strip().upper()] = word[1].strip()
        return arpadict

    def _replace_words(self, input, dictionary):
        regex = re.findall(r"[\w'-]+|[^\w'-]", input)
        assert input == "".join(regex)
        for i in range(len(regex)):
            word = regex[i].upper()
            if word in dictionary.keys():
                regex[i] = "{" + dictionary[word] + "}"
        return "".join(regex)

    def _arpa_parse(self, input):
        z = []
        space = self.parser.labels.index(" ")
        input = self._replace_words(input, self.arpadict)
        input = input.replace("\n", " \n")
        input = input.replace(".", ". ")
        while "{" in input:
            if "}" not in input:
                input.replace("{", "")
            else:
                pre = input[: input.find("{")]
                if pre.strip() != "":
                    x = (
                        torch.tensor(self.parser.encode(pre.strip()))
                        .long()
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    seq_ids = x.squeeze(0).cpu().detach().numpy()
                    z.extend(seq_ids)
                z.append(space)

                arpaword = input[input.find("{") + 1 : input.find("}")]
                arpaword = (
                    arpaword.replace("0", "")
                    .replace("1", "")
                    .replace("2", "")
                    .strip()
                    .split(" ")
                )

                seq_ids = []
                for x in arpaword:
                    if x == "":
                        continue
                    if x.replace("_", " ") not in self.parser.labels:
                        continue
                    seq_ids.append(self.parser.labels.index(x.replace("_", " ")))
                seq_ids.append(space)
                z.extend(seq_ids)
                input = input[input.find("}") + 1 :]
        if input != "":
            x = (
                torch.tensor(self.parser.encode(input.strip()))
                .long()
                .unsqueeze(0)
                .to(self.device)
            )
            seq_ids = x.squeeze(0).cpu().detach().numpy()
            z.extend(seq_ids)
        if z[-1] == space:
            z = z[:-1]
        if z[0] == space:
            z = z[1:]
        return [
            z[i]
            for i in range(len(z))
            if (i == 0) or (z[i] != z[i - 1]) or (z[i] != space)
        ]

    def _to_arpa(self, input):
        arpa = ""
        z = []
        space = self.parser.labels.index(" ")
        while space in input:
            z.append(input[: input.index(space)])
            input = input[input.index(space) + 1 :]
        z.append(input)
        for y in z:
            if len(y) == 0:
                continue

            arpaword = " {"
            for s in y:
                if self.parser.labels[s] == " ":
                    arpaword += "_ "
                else:
                    arpaword += self.parser.labels[s] + " "
            arpaword += "} "
            if (
                not arpaword.replace("{", "")
                .replace("}", "")
                .replace(" ", "")
                .isalnum()
            ):
                arpaword = arpaword.replace("{", "").replace(" }", "")
            arpa += arpaword
        return arpa.replace("  ", " ").replace(" }", "}").strip()

    def _generate_json(self, input, outpath):
        output = ""
        sample_rate = 22050
        lpath = input.split("|")[0].strip()
        size = os.stat(lpath).st_size
        x = {
            "audio_filepath": lpath,
            "duration": size / (sample_rate * 2),
            "text": input.split("|")[1].strip(),
        }
        output += json.dumps(x) + "\n"
        with open(outpath, "w", encoding="utf8") as w:
            w.write(output)

    def _preprocess_tokens(self, tokens, blank):
        new_tokens = [blank]
        for c in tokens:
            new_tokens.extend([c, blank])
        tokens = new_tokens
        return tokens

    def get_tokens(self, transcript):
        token_list = self._arpa_parse(transcript)
        tokens = torch.IntTensor(token_list).view(1, -1).to(self.device)
        arpa = self._to_arpa(token_list)
        return token_list, tokens, arpa

    def get_duration(self, wav_name, transcript, tokens):
        if not os.path.exists(os.path.join(self.run_path, "temp")):
            os.mkdir(os.path.join(self.run_path, "temp"))
        if "_" not in transcript:
            self._generate_json(
                os.path.join(self.run_path, "temp", wav_name + "_conv.wav")
                + "|"
                + transcript.strip(),
                os.path.join(self.run_path, "temp", wav_name + ".json"),
            )
        else:
            self._generate_json(
                os.path.join(self.run_path, "temp", wav_name + "_conv.wav")
                + "|"
                + "dummy",
                os.path.join(self.run_path, "temp", wav_name + ".json"),
            )

        data_config = {
            "manifest_filepath": os.path.join(
                self.run_path, "temp", wav_name + ".json"
            ),
            "sample_rate": 22050,
            "batch_size": 1,
        }

        dataset = nemo.collections.asr.data.audio_to_text._AudioTextDataset(
            manifest_filepath=data_config["manifest_filepath"],
            sample_rate=data_config["sample_rate"],
            parser=self.parser,
        )

        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=data_config["batch_size"],
            collate_fn=dataset.collate_fn,
            shuffle=False,
        )

        blank_id = self.asr_model.decoder.num_classes_with_blank - 1

        for sample_idx, test_sample in tqdm(enumerate(dl), total=len(dl)):
            log_probs, _, greedy_predictions = self.asr_model(
                input_signal=test_sample[0], input_signal_length=test_sample[1]
            )

            log_probs = log_probs[0].cpu().detach().numpy()
            target_tokens = self._preprocess_tokens(tokens, blank_id)

            f, p = forward_extractor(target_tokens, log_probs, blank_id)
            durs = backward_extractor(f, p)

            del test_sample
            return durs
        return None


class ExtractPitch:
    def __init__(self):
        self.spice = None

    def _crepe_f0(self, wav_path, hop_length=256):
        sr, audio = wavfile.read(wav_path)
        audio_x = np.arange(0, len(audio)) / 22050.0
        f0time, frequency, confidence, activation = crepe.predict(
            audio, sr, viterbi=True
        )

        x = np.arange(0, len(audio), hop_length) / 22050.0
        freq_interp = np.interp(x, f0time, frequency)
        freq_interp_nothreshold = np.interp(x, f0time, frequency)
        conf_interp = np.interp(x, f0time, confidence)
        audio_interp = np.interp(x, audio_x, np.absolute(audio)) / 32768.0
        weights = [0.5, 0.25, 0.25]
        audio_smooth = np.convolve(audio_interp, np.array(weights)[::-1], "same")

        conf_threshold = 0.25
        audio_threshold = 0.0005
        for i in range(len(freq_interp)):
            if conf_interp[i] < conf_threshold:
                freq_interp[i] = 0.0
            if audio_smooth[i] < audio_threshold:
                freq_interp[i] = 0.0

        # Hack to make f0 and mel lengths equal
        if len(audio) % hop_length == 0:
            freq_interp = np.pad(freq_interp, pad_width=[0, 1])
            conf_interp = np.pad(conf_interp, pad_width=[0, 1])
        return (
            torch.from_numpy(freq_interp.astype(np.float32)),
            torch.from_numpy(freq_interp_nothreshold.astype(np.float32)),
            torch.from_numpy(conf_interp.astype(np.float32)),
            torch.from_numpy(frequency.astype(np.float32)),
        )

    def _spice_f0(self, wav_path, hop_length=256):
        if self.spice is None:
            self.spice = hub.load("https://tfhub.dev/google/spice/2")

        def output2hz(pitch_output):
            # Constants taken from https://tfhub.dev/google/spice/2
            PT_OFFSET = 25.58
            PT_SLOPE = 63.07
            FMIN = 10.0
            BINS_PER_OCTAVE = 12.0
            cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET
            return FMIN * 2.0 ** (1.0 * cqt_bin / BINS_PER_OCTAVE)

        # Resampling
        sr, audio = wavfile.read(wav_path)
        audio_x = np.arange(0, len(audio)) / 22050.0
        wave = resampy.resample(
            audio,
            sr,
            16000,
            filter="sinc_window",
            window=scipy.signal.windows.hann,
            num_zeros=8,
        ).astype(np.float32)

        # Prediction
        output = self.spice.signatures["serving_default"](tf.constant(wave))
        frequency = [output2hz(p) for p in output["pitch"]]
        confidence = 1.0 - output["uncertainty"]

        # Interpolation
        x = np.arange(0, len(audio), hop_length) / 22050.0
        xp = np.arange(0, len(frequency)) * 0.032
        xp_threshold = []
        fp_threshold = []
        conf_threshold = 0.25
        for i in range(len(xp)):
            if confidence[i] >= conf_threshold:
                xp_threshold.append(xp[i])
                fp_threshold.append(frequency[i])
        freq_zeroes = np.interp(x, xp, frequency)
        freq_threshold = np.interp(x, xp_threshold, fp_threshold)
        conf_interp = np.interp(x, xp, confidence)
        audio_interp = np.interp(x, audio_x, np.absolute(audio)) / 32768.0
        weights = [0.5, 0.25, 0.25]
        audio_smooth = np.convolve(audio_interp, np.array(weights)[::-1], "same")

        audio_threshold = 0.0005
        for i in range(len(freq_zeroes)):
            if conf_interp[i] < conf_threshold:
                freq_zeroes[i] = 0.0
            if audio_smooth[i] < audio_threshold:
                freq_zeroes[i] = 0.0

        # Hack to make f0 and mel lengths equal
        if len(audio) % hop_length == 0:
            freq_zeroes = np.pad(freq_zeroes, pad_width=[0, 1])
            freq_threshold = np.pad(freq_threshold, pad_width=[0, 1])
            conf_interp = np.pad(conf_interp, pad_width=[0, 1])

        return (
            torch.from_numpy(freq_zeroes.astype(np.float32)),
            torch.from_numpy(freq_threshold.astype(np.float32)),
            torch.from_numpy(conf_interp.astype(np.float32)),
        )

    def _torchcrepe_f0(self, wav_path, hop_length=256):
        audio, sr = torchcrepe.load.audio(wav_path)
        audio_numpy = audio.squeeze(0).numpy()
        audio_x = np.arange(0, len(audio_numpy)) / 22050.0

        frequency, confidence = torchcrepe.predict(
            audio,
            sr,
            hop_length=256,
            fmin=50,
            fmax=800,
            model="full",
            decoder=torchcrepe.decode.viterbi,
            return_periodicity=True,
            batch_size=128,
            device="cuda:0",
        )

        x = np.arange(0, len(audio_numpy), hop_length) / 22050.0
        orig_frequency = frequency.squeeze(0).numpy()[: len(x)]
        frequency = frequency.squeeze(0).numpy()[: len(x)]
        confidence = confidence.squeeze(0).numpy()[: len(x)]
        audio_interp = np.interp(x, audio_x, np.absolute(audio_numpy))
        weights = [0.5, 0.25, 0.25]
        audio_smooth = np.convolve(audio_interp, np.array(weights)[::-1], "same")
        conf_smooth = np.convolve(confidence, np.array(weights)[::-1], "same")

        conf_threshold = 0.04
        audio_threshold = 0.0005
        for i in range(len(frequency)):
            if conf_smooth[i] < conf_threshold:
                frequency[i] = 0.0
            if audio_smooth[i] < audio_threshold:
                frequency[i] = 0.0

        # Hack to make f0 and mel lengths equal
        if len(audio_numpy) % hop_length == 0:
            frequency = np.pad(frequency, pad_width=[0, 1])
            orig_frequency = np.pad(frequency, pad_width=[0, 1])
        return (
            torch.from_numpy(frequency.astype(np.float32)),
            torch.from_numpy(orig_frequency.astype(np.float32)),
        )

    def f0_to_audio(self, f0s):
        volume = 0.2
        sr = 22050
        freq = 440.0
        base_audio = (
            np.sin(2 * np.pi * np.arange(256.0 * len(f0s)) * freq / sr) * volume
        ).astype(np.float32)
        shifted_audio = psola.vocode(base_audio, sr, target_pitch=f0s)
        for i in range(len(f0s)):
            if f0s[i] == 0.0:
                shifted_audio[i * 256 : (i + 1) * 256] = 0.0
        print(type(shifted_audio[0]))
        buffer = io.BytesIO()
        wavfile.write(buffer, sr, shifted_audio.astype(np.float32))
        b64 = base64.b64encode(buffer.getvalue())
        sound = "data:audio/x-wav;base64," + b64.decode("ascii")
        return sound

    def get_pitch(self, wav_path, legacy=True):
        if USE_SPICE:
            crepe_zeroes, crepe_nozeroes, crepe_confidence, crepe_raw = self._crepe_f0(
                wav_path
            )
            spice_zeroes, spice_threshold, spice_confidence = self._spice_f0(wav_path)
            for i in range(len(spice_threshold)):
                if crepe_zeroes[i] == 0.0:
                    spice_threshold[i] = 0.0
            return spice_threshold, crepe_raw
        else:
            torchcrepe_zeroes, torchcrepe_nozeroes = self._torchcrepe_f0(wav_path)
            if legacy:
                crepe_zeroes, _, _, crepe_raw = self._crepe_f0(wav_path)
                for i in range(len(crepe_zeroes)):
                    if crepe_zeroes[i] != 0.0:
                        crepe_zeroes[i] = torchcrepe_nozeroes[i]
                return crepe_zeroes, torchcrepe_nozeroes
            else:
                return torchcrepe_zeroes, torchcrepe_nozeroes

    def auto_tune(self, audio_np, audio_torch, f0s_wo_silence):
        output_freq = torchcrepe.predict(
            audio_torch / 32768.0,
            22050,
            hop_length=256,
            fmin=50,
            fmax=800,
            model="full",
            decoder=torchcrepe.decode.viterbi,
            return_periodicity=False,
            batch_size=128,
            device="cuda:0",
        )
        output_freq = output_freq.squeeze(0).cpu().numpy()[: len(f0s_wo_silence)]
        output_pitch = torch.from_numpy(output_freq.astype(np.float32))
        target_pitch = torch.FloatTensor(f0s_wo_silence)
        factor = torch.mean(output_pitch) / torch.mean(target_pitch)

        octaves = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        nearest_octave = min(octaves, key=lambda x: abs(x - factor))
        target_pitch *= nearest_octave
        if len(target_pitch) < len(output_pitch):
            target_pitch = torch.nn.functional.pad(
                target_pitch,
                (0, list(output_pitch.shape)[0] - list(target_pitch.shape)[0]),
                "constant",
                0,
            )
        if len(target_pitch) > len(output_pitch):
            target_pitch = target_pitch[0 : list(output_pitch.shape)[0]]

        audio_np = psola.vocode(audio_np, 22050, target_pitch=target_pitch).astype(
            np.float32
        )
        normalize = (1.0 / np.max(np.abs(audio_np))) ** 0.9
        audio_np = audio_np * normalize * 32768.0
        audio_np = audio_np.astype(np.int16)
        return audio_np

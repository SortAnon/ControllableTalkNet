import os
from flask import Flask, request, send_file
import sys
import torch
import numpy as np
from scipy.io import wavfile
import io
from nemo.collections.tts.models import TalkNetSpectModel
from nemo.collections.tts.models import TalkNetPitchModel
from nemo.collections.tts.models import TalkNetDursModel
import json

sys.path.append("hifi-gan")
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator
from denoiser import Denoiser


app = Flask(__name__)
RUN_PATH = os.path.dirname(os.path.realpath(__file__))
DEVICE = "cuda:0"


def load_hifigan(model_name, conf_name):
    # Load HiFi-GAN
    conf = os.path.join("hifi-gan", conf_name + ".json")
    with open(conf) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan = Generator(h).to(torch.device(DEVICE))
    state_dict_g = torch.load(model_name, map_location=torch.device(DEVICE))
    hifigan.load_state_dict(state_dict_g["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = Denoiser(hifigan, mode="normal")
    return hifigan, h, denoiser


def generate_json(input, outpath):
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


def load_talknet(talknet_path):
    with torch.no_grad():
        tnmodel = TalkNetSpectModel.restore_from(talknet_path)
        durs_path = os.path.join(os.path.dirname(talknet_path), "TalkNetDurs.nemo")
        tndurs = TalkNetDursModel.restore_from(durs_path)
        tnmodel.add_module("_durs_model", tndurs)
        pitch_path = os.path.join(os.path.dirname(talknet_path), "TalkNetPitch.nemo")
        tnpitch = TalkNetPitchModel.restore_from(pitch_path)
        tnmodel.add_module("_pitch_model", tnpitch)
        tnmodel.eval()
        return tnmodel


def generate_audio(transcript, tnmodel, hifigan, denoiser):
    with torch.no_grad():
        tokens = tnmodel.parse(text=transcript.strip())
        spect = tnmodel.generate_spectrogram(tokens=tokens)

        y_g_hat = hifigan(spect.float())
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
        audio_np = audio_denoised.detach().cpu().numpy().reshape(-1).astype(np.int16)

        buffer = io.BytesIO()
        wavfile.write(buffer, 22050, audio_np)
        return buffer


@app.route("/", methods=["GET"])
def get_check():
    return "TalkNet server online"


@app.route("/api/tts", methods=["GET"])
def get_tts():
    if "text" not in request.args:
        return ""
    transcript = request.args.get("text")
    return send_file(
        generate_audio(transcript, tnmodel, hifigan, denoiser),
        attachment_filename="audio.wav",
        mimetype="audio/x-wav",
    )


if __name__ == "__main__":
    hifigan, h, denoiser = load_hifigan(
        os.path.join(
            RUN_PATH, "models", "1QnOliOAmerMUNuo2wXoH-YoainoSjZen", "hifiganmodel"
        ),
        "config_v1",
    )
    tnmodel = load_talknet(
        os.path.join(
            RUN_PATH, "models", "1QnOliOAmerMUNuo2wXoH-YoainoSjZen", "TalkNetSpect.nemo"
        )
    )
    app.run(debug=False)

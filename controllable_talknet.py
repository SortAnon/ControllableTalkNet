import sys
import os
import base64
import dash
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
import torch
import numpy as np
import crepe
import scipy
from scipy.io import wavfile
import psola
import io
import nemo
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.tts.models import TalkNetSpectModel
import json
from tqdm import tqdm
import gdown
import zipfile
import resampy
import traceback
import ffmpeg

sys.path.append("hifi-gan")
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import Generator
from denoiser import Denoiser

app = JupyterDash(__name__)
UPLOAD_DIRECTORY = "/content"

app.layout = html.Div(
    children=[
        html.H1(
            children="Controllable TalkNet",
            style={
                "font-family": "EquestriaWebfont",
                "color": "#ed3c96",
                "font-size": "4em",
                "text-align": "center",
                "margin-top": "0em",
                "margin-bottom": "0em",
            },
        ),
        html.Label("Character selection", htmlFor="model-dropdown"),
        dcc.Dropdown(
            id="model-dropdown",
            options=[
                {
                    "label": "Twilight (singing)",
                    "value": "1DEVEDfYegu1ffPtuGBX6cBUHPfzElY8C",
                },
                {
                    "label": "Twilight (talking)",
                    "value": "1JukWe5hVIFOb4Df20bRi0pMSvlAIE2Jt",
                },
                {
                    "label": "Pinkie (singing)",
                    "value": "19cdMqNJJUFFkurUgYG8ISqr7VLc_6_Co",
                },
                {"label": "Custom model", "value": "Custom"},
            ],
            value=None,
            style={
                "max-width": "90vw",
                "width": "20em",
                "margin-bottom": "0.7em",
            },
        ),
        html.Div(
            children=[
                dcc.Input(
                    id="drive-id",
                    type="text",
                    placeholder="Drive ID for custom model",
                    style={"width": "22em"},
                ),
            ],
            id="custom-model",
            style={
                "display": "none",
            },
        ),
        html.Label(
            "Upload reference audio (22KHz mono WAV) to " + UPLOAD_DIRECTORY,
            htmlFor="reference-dropdown",
        ),
        dcc.Store(id="current-f0s"),
        dcc.Store(id="current-filename"),
        dcc.Loading(
            id="audio-loading",
            children=[
                html.Div(
                    [
                        html.Button(
                            "Update file list",
                            id="update-button",
                            style={
                                "margin-right": "10px",
                            },
                        ),
                        dcc.Dropdown(
                            id="reference-dropdown",
                            options=[],
                            value=None,
                            style={
                                "max-width": "80vw",
                                "width": "30em",
                            },
                        ),
                        dcc.Store(id="pitch-clicks"),
                        html.Button(
                            "Debug pitch",
                            id="pitch-button",
                            style={
                                "margin-left": "10px",
                            },
                        ),
                    ],
                    style={
                        "width": "100%",
                        "display": "flex",
                        "align-items": "center",
                        "justify-content": "center",
                        "flex-direction": "row",
                        "margin-left": "50px",
                        "vertical-align": "middle",
                    },
                ),
                html.Audio(
                    id="pitch-out",
                    controls=True,
                    style={"display": "none"},
                ),
                html.Div(
                    id="audio-loading-output",
                    style={
                        "font-style": "italic",
                        "margin-bottom": "0.7em",
                        "text-align": "center",
                    },
                ),
            ],
            type="default",
        ),
        html.Div(
            [
                dcc.Checklist(
                    id="pitch-options",
                    options=[
                        {"label": "Pitch correction", "value": "pc"},
                        {"label": "Override pitch factor", "value": "pf"},
                    ],
                    value=["pc"],
                ),
                dcc.Input(
                    id="pitch-factor",
                    type="number",
                    value="1.0",
                    style={"width": "7em", "margin-left": "10px"},
                    min=0.1,
                    max=10.0,
                    step=0.01,
                    disabled=True,
                ),
            ],
            style={
                "width": "100%",
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
                "flex-direction": "row",
                "margin-left": "50px",
                "margin-bottom": "0.7em",
            },
        ),
        html.Label("Transcript", htmlFor="transcript-input"),
        dcc.Textarea(
            id="transcript-input",
            value="",
            style={
                "max-width": "90vw",
                "width": "50em",
                "height": "8em",
                "margin-bottom": "0.7em",
            },
        ),
        dcc.Loading(
            html.Div(
                [
                    html.Button(
                        "Generate",
                        id="gen-button",
                    ),
                    html.Audio(
                        id="audio-out",
                        controls=True,
                        style={
                            "display": "none",
                        },
                    ),
                    html.Div(
                        id="generated-info",
                        style={
                            "font-style": "italic",
                        },
                    ),
                ],
                style={
                    "width": "100%",
                    "display": "flex",
                    "align-items": "center",
                    "justify-content": "center",
                    "flex-direction": "column",
                },
            )
        ),
        html.Footer(
            children="""
                Presented by the Pony Preservation Project.
            """,
            style={"margin-top": "2em", "font-size": "0.7em"},
        ),
    ],
    style={
        "width": "100%",
        "display": "flex",
        "align-items": "center",
        "justify-content": "center",
        "flex-direction": "column",
        "background-color": "#FFF",
    },
)


def load_hifigan(model_name, conf_name):
    # Load HiFi-GAN
    conf = os.path.join("hifi-gan", conf_name + ".json")
    with open(conf) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan = Generator(h).to(torch.device("cuda"))
    state_dict_g = torch.load(model_name, map_location=torch.device("cuda"))
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


asr_model = (
    EncDecCTCModel.from_pretrained(model_name="asr_talknet_aligner").cpu().eval()
)


def forward_extractor(tokens, log_probs, blank):
    """Computes states f and p."""
    n, m = len(tokens), log_probs.shape[0]
    # `f[s, t]` -- max sum of log probs for `s` first codes
    # with `t` first timesteps with ending in `tokens[s]`.
    f = np.empty((n + 1, m + 1), dtype=float)
    f.fill(-(10 ** 9))
    p = np.empty((n + 1, m + 1), dtype=int)
    f[0, 0] = 0.0  # Start
    for s in range(1, n + 1):
        c = tokens[s - 1]
        for t in range((s + 1) // 2, m + 1):
            f[s, t] = log_probs[t - 1, c]
            # Option #1: prev char is equal to current one.
            if s == 1 or c == blank or c == tokens[s - 3]:
                options = f[s : (s - 2 if s > 1 else None) : -1, t - 1]
            else:  # Is not equal to current one.
                options = f[s : (s - 3 if s > 2 else None) : -1, t - 1]
            f[s, t] += np.max(options)
            p[s, t] = np.argmax(options)
    return f, p


def backward_extractor(f, p):
    """Computes durs from f and p."""
    n, m = f.shape
    n -= 1
    m -= 1
    durs = np.zeros(n, dtype=int)
    if f[-1, -1] >= f[-2, -1]:
        s, t = n, m
    else:
        s, t = n - 1, m
    while s > 0:
        durs[s - 1] += 1
        s -= p[s, t]
        t -= 1
    assert durs.shape[0] == n
    assert np.sum(durs) == m
    assert np.all(durs[1::2] > 0)
    return durs


def preprocess_tokens(tokens, blank):
    new_tokens = [blank]
    for c in tokens:
        new_tokens.extend([c, blank])
    tokens = new_tokens
    return tokens


def get_duration(wav_name, transcript):
    if not os.path.exists(os.path.join(UPLOAD_DIRECTORY, "output")):
        os.mkdir(os.path.join(UPLOAD_DIRECTORY, "output"))
    if "_" not in transcript:
        generate_json(
            os.path.join(UPLOAD_DIRECTORY, "output", wav_name + "_conv.wav")
            + "|"
            + transcript.strip(),
            os.path.join(UPLOAD_DIRECTORY, "output", wav_name + ".json"),
        )
    else:
        generate_json(
            os.path.join(UPLOAD_DIRECTORY, "output", wav_name + "_conv.wav")
            + "|"
            + "dummy",
            os.path.join(UPLOAD_DIRECTORY, "output", wav_name + ".json"),
        )

    data_config = {
        "manifest_filepath": os.path.join(
            UPLOAD_DIRECTORY, "output", wav_name + ".json"
        ),
        "sample_rate": 22050,
        "batch_size": 1,
    }

    parser = (
        nemo.collections.asr.data.audio_to_text.AudioToCharWithDursF0Dataset.make_vocab(
            notation="phonemes",
            punct=True,
            spaces=True,
            stresses=False,
            add_blank_at="last",
        )
    )

    dataset = nemo.collections.asr.data.audio_to_text._AudioTextDataset(
        manifest_filepath=data_config["manifest_filepath"],
        sample_rate=data_config["sample_rate"],
        parser=parser,
    )

    dl = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=data_config["batch_size"],
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )

    blank_id = asr_model.decoder.num_classes_with_blank - 1

    for sample_idx, test_sample in tqdm(enumerate(dl), total=len(dl)):
        log_probs, _, greedy_predictions = asr_model(
            input_signal=test_sample[0], input_signal_length=test_sample[1]
        )

        log_probs = log_probs[0].cpu().detach().numpy()
        if "_" not in transcript:
            seq_ids = test_sample[2][0].cpu().detach().numpy()
        else:
            pass
            """arpa_input = (
                transcript.replace("0", "")
                .replace("1", "")
                .replace("2", "")
                .replace("_", " _ ")
                .strip()
                .split(" ")
            )

            seq_ids = []
            for x in arpa_input:
                if x == "":
                    continue
                if x.replace("_", " ") not in parser.labels:
                    continue
                seq_ids.append(parser.labels.index(x.replace("_", " ")))"""

        seq_ids = test_sample[2][0].cpu().detach().numpy()
        target_tokens = preprocess_tokens(seq_ids, blank_id)

        f, p = forward_extractor(target_tokens, log_probs, blank_id)
        durs = backward_extractor(f, p)

        arpa = ""
        for s in seq_ids:
            if parser.labels[s] == " ":
                arpa += "_ "
            else:
                arpa += parser.labels[s] + " "

        del test_sample
        return durs, arpa.strip(), seq_ids
    return None, None, None


def crepe_f0(wav_path, hop_length=256):
    # sr, audio = wavfile.read(io.BytesIO(wav_data))
    sr, audio = wavfile.read(wav_path)
    audio_x = np.arange(0, len(audio)) / 22050.0
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)

    x = np.arange(0, len(audio), hop_length) / 22050.0
    freq_interp = np.interp(x, time, frequency)
    conf_interp = np.interp(x, time, confidence)
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
    return torch.from_numpy(freq_interp.astype(np.float32))


def f0_to_audio(f0s):
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


@app.callback(
    dash.dependencies.Output("custom-model", "style"),
    [
        dash.dependencies.Input("model-dropdown", "value"),
    ],
)
def update_model(value):
    if value == "Custom":
        return {"margin-bottom": "0.7em", "display": "block"}
    return {"display": "none"}


@app.callback(
    dash.dependencies.Output("pitch-factor", "disabled"),
    [
        dash.dependencies.Input("pitch-options", "value"),
    ],
)
def update_pitch_options(value):
    return "pf" not in value


playback_style = {
    "margin-top": "0.3em",
    "margin-bottom": "0.3em",
    "display": "block",
    "width": "600px",
    "max-width": "90vw",
}


@app.callback(
    dash.dependencies.Output("reference-dropdown", "options"),
    [
        dash.dependencies.Input("update-button", "n_clicks"),
    ],
)
def update_filelist(n_clicks):
    filelist = []
    supported_formats = [".wav", ".ogg", ".mp3", "flac", ".aac"]
    for x in os.listdir(UPLOAD_DIRECTORY):
        if x[-4:].lower() in supported_formats:
            filelist.append({"label": x, "value": x})
    return filelist


@app.callback(
    [
        dash.dependencies.Output("audio-loading-output", "children"),
        dash.dependencies.Output("current-f0s", "data"),
        dash.dependencies.Output("current-filename", "data"),
    ],
    [
        dash.dependencies.Input("reference-dropdown", "value"),
    ],
)
def select_file(dropdown_value):
    if dropdown_value is not None:
        if not os.path.exists(os.path.join(UPLOAD_DIRECTORY, "output")):
            os.mkdir(os.path.join(UPLOAD_DIRECTORY, "output"))
        ffmpeg.input(os.path.join(UPLOAD_DIRECTORY, dropdown_value)).output(
            os.path.join(UPLOAD_DIRECTORY, "output", dropdown_value + "_conv.wav"),
            ar="22050",
            ac="1",
            acodec="pcm_s16le",
            map_metadata="-1",
            fflags="+bitexact",
        ).overwrite_output().run(quiet=True)
        return [
            "Analyzed " + dropdown_value,
            crepe_f0(
                os.path.join(UPLOAD_DIRECTORY, "output", dropdown_value + "_conv.wav")
            ),
            dropdown_value,
        ]
    else:
        return ["No audio analyzed", None, None]


@app.callback(
    [
        dash.dependencies.Output("pitch-out", "src"),
        dash.dependencies.Output("pitch-out", "style"),
        dash.dependencies.Output("pitch-clicks", "data"),
    ],
    [
        dash.dependencies.Input("pitch-button", "n_clicks"),
        dash.dependencies.Input("pitch-clicks", "data"),
        dash.dependencies.Input("current-f0s", "data"),
    ],
)
def debug_pitch(n_clicks, pitch_clicks, current_f0s):
    if not n_clicks or current_f0s is None or n_clicks <= pitch_clicks:
        if n_clicks is not None:
            pitch_clicks = n_clicks
        else:
            pitch_clicks = 0
        return [
            None,
            {
                "display": "none",
            },
            pitch_clicks,
        ]
    pitch_clicks = n_clicks
    return [f0_to_audio(current_f0s), playback_style, pitch_clicks]


def download_model(model, custom_model):
    global hifigan_sr, h2, denoiser_sr
    d = "https://drive.google.com/uc?id="
    if model == "Custom":
        drive_id = custom_model
    else:
        drive_id = model
    if not os.path.exists(os.path.join(UPLOAD_DIRECTORY, drive_id)):
        os.mkdir(os.path.join(UPLOAD_DIRECTORY, drive_id))

        gdown.download(
            d + drive_id,
            os.path.join(UPLOAD_DIRECTORY, drive_id, "model.zip"),
            quiet=False,
        )
        with zipfile.ZipFile(
            os.path.join(UPLOAD_DIRECTORY, drive_id, "model.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(os.path.join(UPLOAD_DIRECTORY, drive_id))
        os.remove(os.path.join(UPLOAD_DIRECTORY, drive_id, "model.zip"))

    # Download super-resolution HiFi-GAN
    sr_path = "hifi-gan/hifisr"
    if not os.path.exists(sr_path):
        gdown.download(d + "14fOprFAIlCQkVRxsfInhEPG0n-xN4QOa", sr_path, quiet=False)
    if not os.path.exists(sr_path):
        raise Exception("HiFI-GAN model failed to download!")
    hifigan_sr, h2, denoiser_sr = load_hifigan(sr_path, "config_32k")

    return (
        None,
        os.path.join(UPLOAD_DIRECTORY, drive_id, "TalkNetSpect.nemo"),
        os.path.join(UPLOAD_DIRECTORY, drive_id, "hifiganmodel"),
    )


tnmodel, tnpath = None, None
hifigan, h, denoiser, hifipath = None, None, None, None


@app.callback(
    [
        dash.dependencies.Output("audio-out", "src"),
        dash.dependencies.Output("generated-info", "children"),
        dash.dependencies.Output("audio-out", "style"),
        dash.dependencies.Output("pitch-factor", "value"),
    ],
    [dash.dependencies.Input("gen-button", "n_clicks")],
    [
        dash.dependencies.State("model-dropdown", "value"),
        dash.dependencies.State("drive-id", "value"),
        dash.dependencies.State("transcript-input", "value"),
        dash.dependencies.State("pitch-options", "value"),
        dash.dependencies.State("pitch-factor", "value"),
        dash.dependencies.State("current-filename", "data"),
        dash.dependencies.State("current-f0s", "data"),
    ],
)
def generate_audio(
    n_clicks,
    model,
    custom_model,
    transcript,
    pitch_options,
    pitch_factor,
    wav_name,
    f0s,
):
    global tnmodel, tnpath, hifigan, h, denoiser, hifipath

    if n_clicks is None:
        raise PreventUpdate
    if model is None:
        return [
            None,
            "No character selected",
            {
                "display": "none",
            },
            pitch_factor,
        ]
    if transcript is None or transcript.strip() == "":
        return [
            None,
            "No transcript entered",
            {
                "display": "none",
            },
            pitch_factor,
        ]
    if wav_name is None:
        return [
            None,
            "No reference audio uploaded",
            {
                "display": "none",
            },
            pitch_factor,
        ]
    load_error, talknet_path, hifigan_path = download_model(model, custom_model)
    if load_error is not None:
        return [
            None,
            load_error,
            {
                "display": "none",
            },
            pitch_factor,
        ]

    try:
        with torch.no_grad():
            durs, arpa, t = get_duration(wav_name, transcript)
            if tnpath != talknet_path:
                tnmodel = TalkNetSpectModel.restore_from(talknet_path)
                tnmodel.eval()

            tokens = tnmodel.parse(text=transcript.strip())
            spect = tnmodel.force_spectrogram(
                tokens=tokens,
                durs=torch.from_numpy(durs).view(1, -1).to("cuda:0"),
                f0=torch.FloatTensor(f0s).view(1, -1).to("cuda:0"),
            )
            np.save(
                os.path.join(UPLOAD_DIRECTORY, "output", wav_name + ".npy"),
                spect.detach().cpu().numpy(),
            )

            if hifipath != hifigan_path:
                hifigan, h, denoiser = load_hifigan(hifigan_path, "config_v1")

            y_g_hat = hifigan(spect.float())
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
            audio_np = (
                audio_denoised.detach().cpu().numpy().reshape(-1).astype(np.int16)
            )

            # Pitch correction
            if "pc" in pitch_options:

                def get_f0(audio, sr):
                    time, frequency, confidence, activation = crepe.predict(
                        audio, sr, viterbi=True
                    )
                    return torch.from_numpy(frequency.astype(np.float32))

                input_pitch = get_f0(audio_np, 22050)
                target_sr, target_audio = wavfile.read(
                    os.path.join(UPLOAD_DIRECTORY, "output", wav_name + "_conv.wav")
                )
                target_pitch = get_f0(target_audio, target_sr)
                factor = torch.mean(input_pitch) / torch.mean(target_pitch)
                if (
                    max(factor, float(pitch_factor)) / min(factor, float(pitch_factor))
                    < 2.0
                    and float(pitch_factor) != 1.0
                ):
                    factor = float(pitch_factor)
                if "pf" in pitch_options:
                    factor = float(pitch_factor)
                    target_pitch *= factor
                else:
                    octaves = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
                    nearest_octave = min(octaves, key=lambda x: abs(x - factor))
                    target_pitch *= nearest_octave
                if len(target_pitch) < len(input_pitch):
                    target_pitch = torch.nn.functional.pad(
                        target_pitch,
                        (0, list(input_pitch.shape)[0] - list(target_pitch.shape)[0]),
                        "constant",
                        0,
                    )
                if len(target_pitch) > len(input_pitch):
                    target_pitch = target_pitch[0 : list(input_pitch.shape)[0]]

                audio_np = psola.vocode(
                    audio_np, 22050, target_pitch=target_pitch
                ).astype(np.float32)
                normalize = (1.0 / np.max(np.abs(audio_np))) ** 0.9
                audio_np = audio_np * normalize * MAX_WAV_VALUE
                audio_np = audio_np.astype(np.int16)
            else:
                factor = pitch_factor

            # Resample to 32k

            wave = resampy.resample(
                audio_np,
                h.sampling_rate,
                h2.sampling_rate,
                filter="sinc_window",
                window=scipy.signal.windows.hann,
                num_zeros=8,
            )
            wave_out = wave.astype(np.int16)

            # HiFi-GAN super-resolution
            wave = wave / MAX_WAV_VALUE
            wave = torch.FloatTensor(wave).to(torch.device("cuda"))
            new_mel = mel_spectrogram(
                wave.unsqueeze(0),
                h2.n_fft,
                h2.num_mels,
                h2.sampling_rate,
                h2.hop_size,
                h2.win_size,
                h2.fmin,
                h2.fmax,
            )
            y_g_hat2 = hifigan_sr(new_mel)
            audio2 = y_g_hat2.squeeze()
            audio2 = audio2 * MAX_WAV_VALUE
            audio2_denoised = denoiser(audio2.view(1, -1), strength=35)[:, 0]

            # High-pass filter, mixing and denormalizing
            audio2_denoised = audio2_denoised.detach().cpu().numpy().reshape(-1)
            b = scipy.signal.firwin(
                101, cutoff=10500, fs=h2.sampling_rate, pass_zero=False
            )
            y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
            y *= 4.0  # superres strength
            y_out = y.astype(np.int16)
            y_padded = np.zeros(wave_out.shape)
            y_padded[: y_out.shape[0]] = y_out
            sr_mix = wave_out + y_padded

            buffer = io.BytesIO()
            wavfile.write(buffer, 32000, sr_mix.astype(np.int16))
            b64 = base64.b64encode(buffer.getvalue())
            sound = "data:audio/x-wav;base64," + b64.decode("ascii")

            return [sound, arpa, playback_style, factor]
    except Exception:
        return [
            None,
            str(traceback.format_exc()),
            {
                "display": "none",
            },
            pitch_factor,
        ]


if __name__ == "__main__":
    app.run_server(
        mode="external",
        debug=True,
        dev_tools_ui=True,
        dev_tools_hot_reload=True,
        threaded=True,
    )

import sys
import os
import io
import base64
import dash
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
import gdown
import traceback
from scipy.io import wavfile
import numpy as np
import torch

sys.path.append("DiffSVC_inference_only")
from end2end import load_e2e_diffsvc, endtoend_from_path, write_to_file

app = JupyterDash(__name__)
UPLOAD_DIRECTORY = "/content"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app.layout = html.Div(
    children=[
        html.H1(
            children="DiffSVC",
            style={
                "font-family": "EquestriaWebfont",
                "color": "#280e5f",
                "font-size": "4em",
                "text-align": "center",
                "margin-top": "0em",
                "margin-bottom": "0em",
            },
        ),
        html.Label("Character selection", htmlFor="speaker-dropdown"),
        dcc.Dropdown(
            id="speaker-dropdown",
            options=[
                {
                    "label": "Twilight Sparkle",
                    "value": "Twilight",
                },
                {
                    "label": "Discord",
                    "value": "Discord",
                },
                {
                    "label": "Pinkie Pie",
                    "value": "Pinkie",
                },
                {"label": "Nancy", "value": "Nancy"},
            ],
            value=None,
            style={
                "max-width": "90vw",
                "width": "20em",
                "margin-bottom": "0.7em",
            },
        ),
        html.Label(
            "Upload reference audio to " + UPLOAD_DIRECTORY,
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
                    id="svc-options",
                    options=[
                        {"label": "Singing mode", "value": "no_pc"},
                    ],
                    value=[],
                )
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
        html.Div(
            [
                html.Label("Steps", htmlFor="linsteps-input"),
                dcc.Input(
                    id="linsteps-input",
                    type="number",
                    value="1000",
                    style={"width": "7em", "margin-left": "10px"},
                    min=1,
                    max=1000,
                    step=1,
                ),
                html.Label("Step strength", htmlFor="linend-input"),
                dcc.Input(
                    id="linend-input",
                    type="number",
                    value="0.06",
                    style={"width": "7em", "margin-left": "10px"},
                    min=0.01,
                    max=0.3,
                    step=0.01,
                ),
            ],
            style={
                "display": "grid",
                "grid-template-columns": "auto 100px",
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
                Presented by the Pony Preservation Project. Models by Cookie.
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


playback_style = {
    "margin-top": "0.3em",
    "margin-bottom": "0.3em",
    "display": "block",
    "width": "600px",
    "max-width": "90vw",
}

playback_hide = {
    "display": "none",
}


@app.callback(
    dash.dependencies.Output("reference-dropdown", "options"),
    [
        dash.dependencies.Input("update-button", "n_clicks"),
    ],
)
def update_filelist(n_clicks):
    filelist = []
    supported_formats = [".wav", ".ogg", ".mp3", "flac"]
    for x in os.listdir(UPLOAD_DIRECTORY):
        if x[-4:].lower() in supported_formats:
            filelist.append({"label": x, "value": x})
    return filelist


@app.callback(
    [
        dash.dependencies.Output("audio-loading-output", "children"),
        dash.dependencies.Output("current-filename", "data"),
    ],
    [
        dash.dependencies.Input("reference-dropdown", "value"),
    ],
)
def select_file(dropdown_value):
    if dropdown_value is not None:
        return [
            "Selected " + dropdown_value,
            dropdown_value,
        ]
    else:
        return ["No audio selected", None]


def download_model(drive_id, outname):
    d = "https://drive.google.com/uc?id="
    model_dir = os.path.join(UPLOAD_DIRECTORY, "models")
    out_fullpath = os.path.join(model_dir, outname)
    out_dir = os.path.dirname(out_fullpath)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(out_fullpath):
        os.makedirs(out_dir, exist_ok=True)
        gdown.download(
            d + drive_id,
            out_fullpath,
            quiet=False,
        )
    return out_fullpath


svc_model, svc_name, last_end, last_step = None, None, None, None


@app.callback(
    [
        dash.dependencies.Output("audio-out", "src"),
        dash.dependencies.Output("generated-info", "children"),
        dash.dependencies.Output("audio-out", "style"),
    ],
    [dash.dependencies.Input("gen-button", "n_clicks")],
    [
        dash.dependencies.State("speaker-dropdown", "value"),
        dash.dependencies.State("svc-options", "value"),
        dash.dependencies.State("current-filename", "data"),
        dash.dependencies.State("linend-input", "value"),
        dash.dependencies.State("linsteps-input", "value"),
    ],
)
def generate_audio(n_clicks, speaker, svc_options, wav_name, lin_end, lin_n_steps):
    global svc_model, svc_name, last_end, last_step

    if n_clicks is None:
        raise PreventUpdate
    if speaker is None:
        return [None, "No character selected", playback_hide]
    if wav_name is None:
        return [None, "No reference audio selected", playback_hide]

    try:
        diffsvc_id = "1Uh17L1JtynFDgX9X6jKLPtD_t4IcTEp9"
        asr_id = "1qt0pGhCbH0TltFFwSP2Oy3NClthcpr7N"
        hifigan_id = "1QQT0HjMhGgDuPhyesZYbvck2lEY7GHAB"
        hifigan_config_id = "1mi_O54zi6nW2eU6VuAO_Y4gv_Df-H9Yv"

        diffsvc_path = download_model(
            diffsvc_id,
            os.path.join(diffsvc_id, "diffsvc_model"),
        )
        asr_path = download_model(
            asr_id,
            os.path.join(asr_id, "asr_model"),
        )
        hifigan_path = download_model(
            hifigan_id,
            os.path.join(hifigan_id, "hifigan_model"),
        )
        download_model(
            hifigan_config_id,
            os.path.join(hifigan_id, "config.json"),
        )
        if svc_name != diffsvc_id:
            svc_model = load_e2e_diffsvc(
                diffsvc_path=diffsvc_path,
                dilated_asr_path=asr_path,
                hifigan_path=hifigan_path,
                device=DEVICE,
            )
            svc_name = diffsvc_id
        if last_end != lin_end or last_step != lin_n_steps:
            svc_model[0].generator.diffusion.set_noise_schedule(
                1e-4, float(lin_end), int(lin_n_steps), device=DEVICE
            )
            last_end = lin_end
            last_step = lin_n_steps

        pred_audio = endtoend_from_path(
            *svc_model,
            os.path.join(UPLOAD_DIRECTORY, wav_name),
            speaker,
            "no_pc" not in svc_options,
            "no_pc" not in svc_options,
            "no_pc" not in svc_options,
            t_max_step=int(lin_n_steps),
        )
        pred_audio *= 32768.0

        buffer = io.BytesIO()
        wavfile.write(
            buffer,
            svc_model[4].sampling_rate,
            pred_audio.squeeze().cpu().numpy().astype(np.int16),
        )
        b64 = base64.b64encode(buffer.getvalue())
        sound = "data:audio/x-wav;base64," + b64.decode("ascii")
        return [sound, "Conversion complete", playback_style]
    except Exception:
        return [None, str(traceback.format_exc()), playback_hide]


if __name__ == "__main__":
    app.run_server(
        mode="external",
        debug=True,
        dev_tools_ui=True,
        dev_tools_hot_reload=True,
        threaded=True,
    )

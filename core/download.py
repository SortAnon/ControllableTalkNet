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

import os
import zipfile
import gdown


def download_from_drive(model, custom_model, run_path):
    try:
        d = "https://drive.google.com/uc?id="
        if model == "Custom":
            drive_id = custom_model
        else:
            drive_id = model
        if drive_id == "" or drive_id is None:
            return ("Missing Drive ID", None, None)
        if not os.path.exists(os.path.join(run_path, "models")):
            os.mkdir(os.path.join(run_path, "models"))
        if not os.path.exists(os.path.join(run_path, "models", drive_id)):
            os.mkdir(os.path.join(run_path, "models", drive_id))
            zip_path = os.path.join(run_path, "models", drive_id, "model.zip")
            gdown.download(
                d + drive_id,
                zip_path,
                quiet=False,
            )
            if not os.path.exists(zip_path):
                os.rmdir(os.path.join(run_path, "models", drive_id))
                return ("Model download failed", None, None)
            if os.stat(zip_path).st_size < 16:
                os.remove(zip_path)
                os.rmdir(os.path.join(run_path, "models", drive_id))
                return ("Model zip is empty", None, None)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(os.path.join(run_path, "models", drive_id))
            os.remove(zip_path)

        # Download super-resolution HiFi-GAN
        sr_path = os.path.join(run_path, "models", "hifisr")
        if not os.path.exists(sr_path):
            gdown.download(
                d + "14fOprFAIlCQkVRxsfInhEPG0n-xN4QOa", sr_path, quiet=False
            )
        if not os.path.exists(sr_path):
            raise Exception("Super-res model failed to download!")

        # Download VQGAN reconstruction model
        rec_path = os.path.join(
            run_path,
            "models",
            "vqgan32_universal_57000.ckpt",
        )
        if not os.path.exists(rec_path):
            gdown.download(
                d + "1wlilvBtlBiAUEqqdqE0AEqo-UKx2X_cL", rec_path, quiet=False
            )
        if not os.path.exists(rec_path):
            raise Exception("Reconstruction VQGAN failed to download!")

        # Download reconstruction HiFi-GAN
        rec_path = os.path.join(
            run_path,
            "models",
            "hifirec",
        )
        if not os.path.exists(rec_path):
            gdown.download(
                d + "12gRIdg65xWiSScvFUFPT5JoPRsijQN90", rec_path, quiet=False
            )
        if not os.path.exists(rec_path):
            raise Exception("Reconstruction HiFi-GAN failed to download!")

        return (
            None,
            os.path.join(run_path, "models", drive_id, "TalkNetSpect.nemo"),
            os.path.join(run_path, "models", drive_id, "hifiganmodel"),
        )
    except Exception as e:
        return (str(e), None, None)

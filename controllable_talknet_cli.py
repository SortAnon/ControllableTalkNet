# Copyright (C) 2023 HydrusBeta
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import controllable_talknet

import librosa
import soundfile

import base64
import io
import os
import argparse
import hashlib

RESULTS_DIR = os.path.join(controllable_talknet.RUN_PATH, 'results')

def parse_arguments():
    parser = argparse.ArgumentParser(prog='Controllable TalkNet',
                                     description="A text-to-speech program based on NVIDIA's implementation of "
                                                 "Talknet2, with some changes to support singing synthesis and higher "
                                                 "audio quality.")
    parser.add_argument('-t', '--text',            type=str, required=True,                                                                   help='The text you would like a pony to say.')
    parser.add_argument('-i', '--reference_audio', type=str,                                                                                  help='The reference audio file to use for guiding the pacing, pitch, and inflection of the generated voice.')
    parser.add_argument('-c', '--character',       type=str, required=True,             choices=known_characters(),                           help='The name of the pony character whose voice you would like to generate.'),
    parser.add_argument('-f', '--pitch_factor',    type=int,                default=0,                                                        help='An integer specifying how many semitones by which to shift the pitch of the input audio.')
    parser.add_argument('-o', '--output',          type=str,                                                                                  help='The desired output filepath. Be sure to include the desired extension, such as .flac or .mp3. If this argument is not passed, then the output will be written as a flac file to a "results" subdirectory in the ControllableTalkNet directory')
    parser.add_argument('-p', '--pitch_options',   type=str,                default=[], choices=['srec', 'pc'],     nargs=argparse.REMAINDER, help='One or both of the following values: "pc", which instructs Controllable TalkNet to auto-tune the output using the reference audio, and "srec", which instructs Controllable TalkNet to attempt to reduce metallic noise in the output.')
    return parser.parse_args()

def known_characters():
    dropdown_options, _ = controllable_talknet.init_dropdown.__wrapped__(None)
    return [option['label'] for option in dropdown_options]

def amend_pitch_options_if_needed(args):
    # If no reference audio is supplied, add 'dra' (Disable Reference Audio) to pitch_options
    if not args.reference_audio:
        args.pitch_options = list(set(args.pitch_options + ['dra']))

    # If a nonzero pitch shift is specified, add 'pf' (Pitch Factor) to pitch_options
    if args.pitch_factor != 0:
        args.pitch_options = list(set(args.pitch_options + ['pf']))

    return args.pitch_options

def create_unique_file(args):
    # Create a unique file name by hashing all the arguments together. Prepend part of the input text to make it easier
    # for the user to find the output file they just generated. The file will be placed in the results directory;
    # return the full path to the file.
    input_hash = ''
    if args.reference_audio and 'dra' not in args.pitch_options:
        input_data, _ = librosa.load(args.reference_audio, sr=None)
        input_hash = hashlib.sha256(input_data).hexdigest()[:20]
    base_string = args.text + input_hash + args.character + str(args.pitch_factor) + ''.join(args.pitch_options)
    full_hash = hashlib.sha256(base_string.encode('utf-8')).hexdigest()[:20]
    unique_filename = args.text[:15] + ('...' if len(args.text) > 15 else '_') + full_hash + '.flac'
    return os.path.join(RESULTS_DIR, unique_filename)

def prepare_output_directory():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

def generate_audio(args) -> (float, int):
    f0s, f0s_wo_silence, wav_name = None, None, None
    if args.reference_audio is not None:
        _, f0s, f0s_wo_silence, wav_name = controllable_talknet.select_file.__wrapped__(args.reference_audio, [''])
    drive_id = get_drive_id_from_character(args.character)
    src, _, _, _ = controllable_talknet.generate_audio.__wrapped__(0, drive_id, None, args.text, args.pitch_options,
                                                                   args.pitch_factor, wav_name, f0s, f0s_wo_silence)
    return get_audio_from_src(src, encoding='ascii')

def get_drive_id_from_character(character):
    dropdown_options, _ = controllable_talknet.init_dropdown.__wrapped__(None)
    character_to_id_map = {option['label']: option['value'].split('|')[0] for option in dropdown_options}
    return character_to_id_map.get(character)

def write_output_file(args):
    output_path = create_unique_file(args) if args.output is None else args.output
    soundfile.write(output_path, output_array, output_samplerate)

def get_audio_from_src(src, encoding):
    _, raw = src.split(',')
    b64_output_bytes = raw.encode(encoding)
    output_bytes = base64.b64decode(b64_output_bytes)
    buffer = io.BytesIO(output_bytes)
    return librosa.load(buffer, sr=None)

if __name__ == '__main__':
    args = parse_arguments()
    args.pitch_options = amend_pitch_options_if_needed(args)
    prepare_output_directory()
    output_array, output_samplerate = generate_audio(args)
    write_output_file(args)


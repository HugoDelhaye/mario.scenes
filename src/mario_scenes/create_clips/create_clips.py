import argparse
import os
import os.path as op
import gzip
import retro
import pandas as pd
import numpy as np
import skvideo.io
from PIL import Image
from joblib import Parallel, delayed
import json
import logging
import re
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import traceback
from mario_scenes.load_data import load_scenes_info
from mario_replays.utils import replay_bk2, get_variables_from_replay, make_mp4, make_gif, make_webp, create_sidecar_dict
from mario_replays.load_data import collect_bk2_files

def prune_variables(variables, start_idx, end_idx):
    """
    Prune variables to the specified frame range.
    """
    pruned_variables = {}
    for key, value in variables.items():
        if isinstance(value, (list, np.ndarray)):
            pruned_variables[key] = value[start_idx:end_idx]
        else:
            pruned_variables[key] = value
    return pruned_variables

def merge_metadata(metadata, sidecar):
    """
    Merge metadata with sidecar variables.
    """
    enriched_metadata = metadata.copy()
    enriched_metadata.update(sidecar)
    return enriched_metadata


def process_bk2_file(bk2_info, args, scenes_info_dict, DATA_PATH, OUTPUT_FOLDER, STIMULI_PATH):
    """
    Process a single bk2 file to extract clips, saving only the requested file types:
    - savestate (.state)
    - ramdump (.npz)
    - gif (.gif)
    - mp4 (.mp4)
    - webp (.webp)
    - json (.json)
    """
    # Add stimuli path in each child process
    retro.data.Integrations.add_custom_path(STIMULI_PATH)

    error_logs = []
    processing_stats = {
        'bk2_file': bk2_info['bk2_file'],
        'clips_processed': 0,
        'clips_skipped': 0,
        'errors': 0,
    }

    try:
        bk2_file = bk2_info['bk2_file']
        bk2_idx = bk2_info['bk2_idx']
        sub = bk2_info['sub']
        ses = bk2_info['ses']
        run = bk2_info['run']
        skip_first_step = bk2_idx == 0

        logging.info(f"Processing bk2 file: {bk2_file}")
        rep_order_string = f'{str(ses).zfill(3)}{str(run).zfill(2)}{str(bk2_idx).zfill(2)}'
        curr_level = op.basename(bk2_file).split("_")[-2].split('-')[1]

        # If the level doesn't match anything in the scenes dictionary, no scenes to extract
        if not any(curr_level in x for x in scenes_info_dict.keys()):
            logging.info(f"No matching scenes for level {curr_level} in {bk2_file}, skipping.")
            return error_logs, processing_stats

        # Run replay
        repetition_variables, replay_info, frames_list, replay_states = get_variables_from_replay(
            op.join(DATA_PATH, bk2_file),
            skip_first_step=skip_first_step,
            game=args.game_name,
            inttype=retro.data.Integrations.CUSTOM_ONLY
        )
        n_frames_total = len(frames_list)
        repetition_variables['player_x_pos'] = [
            hi * 256 + lo for hi, lo in zip(repetition_variables['player_x_posHi'], repetition_variables['player_x_posLo'])
        ]

        scenes_in_current_level = [x for x in scenes_info_dict.keys() if curr_level in x]
        for current_scene in tqdm(scenes_in_current_level, desc=f"Processing scenes in {bk2_file}", leave=False):
            scenes_info_found = []
            scene_start = scenes_info_dict[current_scene]['start']
            scene_end = scenes_info_dict[current_scene]['end']
            level_layout = scenes_info_dict[current_scene]['level_layout']

            start_found = False
            for frame_idx in range(1, n_frames_total):
                if not start_found:
                    if (
                        repetition_variables['player_x_pos'][frame_idx] >= scene_start
                        and repetition_variables['player_x_pos'][frame_idx - 1] < scene_start
                        and repetition_variables['player_x_pos'][frame_idx] < scene_end
                        and repetition_variables['level_layout'][frame_idx] == level_layout
                    ):
                        start_idx = frame_idx
                        start_found = True
                else:
                    if (
                        (repetition_variables['player_x_pos'][frame_idx] >= scene_end
                         and repetition_variables['player_x_pos'][frame_idx - 1] < scene_end)
                        or (repetition_variables['lives'][frame_idx] - repetition_variables['lives'][frame_idx - 1] < 0)
                    ):
                        end_idx = frame_idx
                        start_found = False
                        scenes_info_found.append([start_idx, end_idx])
                    elif (
                        repetition_variables['player_x_pos'][frame_idx] >= scene_start
                        and repetition_variables['player_x_pos'][frame_idx - 1] < scene_start
                    ):
                        start_idx = frame_idx

            for pattern in scenes_info_found:
                start_idx, end_idx = pattern

                selected_frames = frames_list[start_idx:end_idx]

                clip_code = f'{rep_order_string}{str(start_idx).zfill(7)}'
                assert len(clip_code) == 14, f"Invalid clip code: {clip_code}"

                # Construct BIDS-compliant paths
                deriv_folder = op.join(OUTPUT_FOLDER, args.output_name)
                sub_folder = op.join(deriv_folder, f"sub-{sub}")
                ses_folder = op.join(sub_folder, f"ses-{ses}")
                beh_folder = op.join(ses_folder, 'beh')
                clips_folder = op.join(beh_folder, 'clips')
                savestates_folder = op.join(beh_folder, 'savestates')
                os.makedirs(clips_folder, exist_ok=True)
                os.makedirs(savestates_folder, exist_ok=True)

                entities = (
                    f"sub-{sub}_ses-{ses}_run-{run}_level-{repetition_variables['level']}_"
                    f"scene-{int(current_scene.split('s')[1])}_clip-{clip_code}"
                )

                # Prepare output filenames
                gif_fname       = op.join(clips_folder,      f"{entities}_beh.gif")
                mp4_fname       = op.join(clips_folder,      f"{entities}_beh.mp4")
                webp_fname      = op.join(clips_folder,      f"{entities}_beh.webp")
                savestate_fname = op.join(savestates_folder, f"{entities}_beh.state")
                ramdump_fname   = savestate_fname.replace(".state", "_ramdump.npz")
                json_fname      = op.join(clips_folder,      f"{entities}_beh.json")

                metadata = {
                    'Subject': sub,
                    'Session': ses,
                    'Run': run,
                    'Level': repetition_variables['level'],
                    'Scene': int(current_scene.split('s')[1]),
                    'ClipCode': clip_code,
                    'StartFrame': start_idx,
                    'EndFrame': end_idx,
                    'TotalFrames': n_frames_total,
                    'bk2_filepath': bk2_file,
                    'GameName': args.game_name,
                }

                scene_variables = prune_variables(repetition_variables, start_idx, end_idx)

                scene_sidecar = create_sidecar_dict(scene_variables)

                enriched_metadata = merge_metadata(metadata, scene_sidecar)

                with open(json_fname, 'w') as json_file:
                    json.dump(enriched_metadata, json_file, indent=4)

                # If nothing is needed for this clip, skip it.
                if not any([args.save_states, args.save_ramdumps, args.save_videos]):
                    logging.info(f"All requested files exist for clip code {clip_code}, skipping.")
                    processing_stats['clips_skipped'] += 1
                    continue

                try:
                    # Generate GIF
                    if args.save_videos:
                        if args.video_format == 'gif':
                            make_gif(selected_frames, gif_fname)
                        elif args.video_format == 'mp4':
                            make_mp4(selected_frames, mp4_fname)
                        elif args.video_format == 'webp':
                            make_webp(selected_frames, webp_fname)

                    if args.save_states:
                        with gzip.open(savestate_fname, "wb") as fh:
                            fh.write(replay_states[start_idx])
                    if args.save_ramdumps:
                        np.savez_compressed(ramdump_fname, replay_states[start_idx:end_idx])


                    processing_stats['clips_processed'] += 1

                except Exception as e:
                    error_message = f"Error processing clip {clip_code} in bk2 file {bk2_file}: {str(e)}"
                    error_logs.append(error_message)
                    processing_stats['errors'] += 1
                    continue

    except Exception as e:
        bk2_file = bk2_info.get('bk2_file', 'Unknown file')
        error_message = f"Error processing bk2 file {bk2_file}: {str(e)}"
        error_logs.append(error_message)
        processing_stats['errors'] += 1
        print("Full traceback:")
        traceback.print_exc()
        print("\nError message:")
        print(e)

    return error_logs, processing_stats



def main(args):
    # Get datapath
    DATA_PATH = op.abspath(args.datapath)

    # Set up logging based on verbosity level
    if args.verbose == 0:
        logging_level = logging.WARNING
    elif args.verbose == 1:
        logging_level = logging.INFO
    else:
        logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level, format='%(levelname)s: %(message)s')

    # If user provides --simple, use the simpler NES version
    # and change pipeline folder name accordingly.
    if args.simple:
        args.game_name = 'SuperMarioBrosSimple-Nes'
        args.output_name = 'scene_clips_simple'
    else:
        args.game_name = 'SuperMarioBros-Nes'
        args.output_name = 'scene_clips'


    # Load scenes
    scenes_info_dict = load_scenes_info(format='dict')

    # Setup derivatives folder
    if args.output is None:
        OUTPUT_FOLDER = op.join(DATA_PATH, "derivatives")
    else:
        OUTPUT_FOLDER = op.abspath(args.output)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Integrate game
    if args.stimuli is None:
        STIMULI_PATH = op.abspath(op.join(DATA_PATH, "stimuli"))
    else:
        STIMULI_PATH = op.abspath(args.stimuli_path)

    logging.debug(f"Adding stimuli path: {STIMULI_PATH}")
    retro.data.Integrations.add_custom_path(STIMULI_PATH)
    games_list = retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY)
    logging.debug(f"Available games: {games_list}")
    logging.info(f"Game to use: {args.game_name}")
    logging.info(f"Output dataset name: {args.output_name}")
    logging.info(f"Generating clips for the dataset in: {DATA_PATH}")
    logging.info(f"Taking stimuli from: {STIMULI_PATH}")
    logging.info(f"Saving derivatives in: {OUTPUT_FOLDER}")

    # Collect all bk2 files and related information
    bk2_files_info = collect_bk2_files(DATA_PATH, args.subjects, args.sessions)
    total_bk2_files = len(bk2_files_info)

    # Process bk2 files in parallel with progress bar
    n_jobs = args.n_jobs
    logging.info(f"Processing {total_bk2_files} bk2 files using {n_jobs} job(s)...")

    with tqdm_joblib(tqdm(desc="Processing bk2 files", total=total_bk2_files)):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_bk2_file)(bk2_info, args, scenes_info_dict, DATA_PATH, OUTPUT_FOLDER, STIMULI_PATH)
            for bk2_info in bk2_files_info
        )

    # Initialize aggregators
    total_processing_stats = {
        'total_bk2_files': total_bk2_files,
        'total_clips_processed': 0,
        'total_clips_skipped': 0,
        'total_errors': 0,
    }
    all_error_logs = []

    # Aggregate results
    for error_logs, processing_stats in results:
        total_processing_stats['total_clips_processed'] += processing_stats.get('clips_processed', 0)
        total_processing_stats['total_clips_skipped'] += processing_stats.get('clips_skipped', 0)
        total_processing_stats['total_errors'] += processing_stats.get('errors', 0)
        all_error_logs.extend(error_logs)

    # Prepare data for saving: BIDS derivatives dataset_description
    dataset_description = {
        'Name': args.output_name,
        'BIDSVersion': '1.6.0',
        'GeneratedBy': [{
            'Name': 'Courtois Neuromod',
            'Version': '1.0.0',
            'CodeURL': 'https://github.com/courtois-neuromod/mario.scenes/src/mario_scenes/clips_extraction/clip_extraction.py'
        }],
        'SourceDatasets': [{'URL': 'https://github.com/courtois-neuromod/mario/'}],
        'License': 'CC0',
    }
    deriv_folder = op.join(OUTPUT_FOLDER, args.output_name)
    os.makedirs(deriv_folder, exist_ok=True)
    with open(op.join(deriv_folder, "dataset_description.json"), "w") as f:
        json.dump(dataset_description, f, indent=4)

    # Write error logs to a file
    log_file = op.join(deriv_folder, "processing_log.txt")
    with open(log_file, "w") as f:
        f.write("Processing Log\n")
        f.write("=================\n")
        f.write(f"Total bk2 files: {total_processing_stats['total_bk2_files']}\n")
        f.write(f"Total clips processed: {total_processing_stats['total_clips_processed']}\n")
        f.write(f"Total clips skipped: {total_processing_stats['total_clips_skipped']}\n")
        f.write(f"Total errors: {total_processing_stats['total_errors']}\n")
        f.write("\nError Details:\n")
        for error in all_error_logs:
            f.write(error + "\n")

    logging.info(f"Processing complete. Log file saved to {log_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract clips from Mario dataset based on scene information.")
    parser.add_argument(
        "-d",
        "--datapath",
        default='.',
        type=str,
        help="Data path to look for events.tsv and .bk2 files. Should be the root of the Mario dataset.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="Path to the derivatives folder, where the outputs will be saved.",
    )
    parser.add_argument(
        "-sp",
        "--stimuli",
        default=None,
        type=str,
        help="Path to the stimuli folder containing the game ROMs. Defaults to <datapath>/stimuli if not specified.",
    )
    parser.add_argument(
        "-n",
        "--n_jobs",
        default=-1,
        type=int,
        help="Number of CPU cores to use for parallel processing.",
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save the playback video file (.mp4).",
    )
    parser.add_argument(
        "--save_variables",
        action="store_true",
        help="Save the variables file (.npz) that contains game variables.",
    )
    parser.add_argument(
        "--save_states",
        action="store_true",
        help="Save full RAM state at each frame into a *_states.npy file.",
    )
    parser.add_argument(
        "--save_ramdumps",
        action="store_true",
        help="Save RAM dumps at each frame into a *_ramdumps.npy file.",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="If set, use the simplified game version (SuperMarioBrosSimple-Nes) "
             "and output into 'mario_scenes_simple' subfolder instead of 'mario_scenes'."
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Increase verbosity level (can be specified multiple times)'
    )
    parser.add_argument(
        '--subjects', '-sub', nargs='+', default=None,
        help='List of subjects to process (e.g., sub-01 sub-02). If not specified, all subjects are processed.'
    )
    parser.add_argument(
        '--sessions', '-ses', nargs='+', default=None,
        help='List of sessions to process (e.g., ses-001 ses-002). If not specified, all sessions are processed.'
    )
    parser.add_argument(
        '--video_format', '-vf', default='mp4',
        choices=['gif', 'mp4', 'webp'],
        help='Video format to save (default: mp4).'
    )

    args = parser.parse_args()


    main(args)


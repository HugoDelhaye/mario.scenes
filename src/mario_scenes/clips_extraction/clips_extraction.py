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


def replay_bk2(
    bk2_path, skip_first_step=True, game=None, scenario=None, inttype=retro.data.Integrations.CUSTOM_ONLY
):
    """Create an iterator that replays a bk2 file, yielding frames, keypresses, annotations, sound, actions, and state."""
    movie = retro.Movie(bk2_path)
    if game is None:
        # If no game is specified, take it from the bk2 header
        game = movie.get_game()
    logging.debug(f"Creating emulator for game: {game}")
    emulator = retro.make(game, scenario=scenario, inttype=inttype, render_mode=None)
    emulator.initial_state = movie.get_state()
    actions = emulator.buttons
    emulator.reset()

    if skip_first_step:
        movie.step()

    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(emulator.num_buttons):
                keys.append(movie.get_key(i, p))
        frame, rew, terminate, truncate, info = emulator.step(keys)
        annotations = {"reward": rew, "done": terminate, "info": info}
        state = emulator.em.get_state()
        yield frame, keys, annotations, None, actions, state

    emulator.close()
    movie.close()


def get_variables_from_replay(
    bk2_fpath, skip_first_step=True, game=None, scenario=None, inttype=retro.data.Integrations.CUSTOM_ONLY
):
    """Replay the bk2 file and return game variables and frames."""
    replay = replay_bk2(
        bk2_fpath, skip_first_step=skip_first_step, game=game, scenario=scenario, inttype=inttype
    )
    all_frames = []
    all_keys = []
    all_info = []
    annotations = {}

    for frame, keys, annotations, _, actions, _ in replay:
        all_keys.append(keys)
        all_info.append(annotations["info"])
        all_frames.append(frame)

    repetition_variables = reformat_info(all_info, all_keys, bk2_fpath, actions)

    if not annotations.get('done', False):
        logging.warning(f"Done condition not satisfied for {bk2_fpath}. Consider changing skip_first_step.")

    return repetition_variables, all_frames


def reformat_info(info, keys, bk2_fpath, actions):
    """Create a structured dictionary from replay info."""
    filename = op.basename(bk2_fpath)
    entities = filename.split('_')
    entities_dict = {}
    for ent in entities:
        if '-' in ent:
            key, value = ent.split('-', 1)
            entities_dict[key] = value

    repetition_variables = {
        "filename": bk2_fpath,
        "level": entities_dict.get('level'),
        "subject": entities_dict.get('sub'),
        "session": entities_dict.get('ses'),
        "repetition": entities_dict.get('run'),
        "actions": actions,
    }

    for key in info[0].keys():
        repetition_variables[key] = []
    for button in actions:
        repetition_variables[button] = []

    for frame_idx, frame_info in enumerate(info):
        for key in frame_info.keys():
            repetition_variables[key].append(frame_info[key])
        for button_idx, button in enumerate(actions):
            repetition_variables[button].append(keys[frame_idx][button_idx])

    return repetition_variables


def make_gif(selected_frames, movie_fname):
    """Create a GIF file from a list of frames."""
    frame_list = [Image.fromarray(np.uint8(img), "RGB") for img in selected_frames]

    if not frame_list:
        logging.warning(f"No frames to save in {movie_fname}")
        return

    frame_list[0].save(
        movie_fname, save_all=True, append_images=frame_list[1:], optimize=False, duration=16, loop=0
    )


def make_mp4(selected_frames, movie_fname):
    """Create an MP4 file from a list of frames."""
    writer = skvideo.io.FFmpegWriter(
        movie_fname, inputdict={"-r": "60"}, outputdict={"-r": "60"}
    )
    for frame in selected_frames:
        im = Image.new("RGB", (frame.shape[1], frame.shape[0]), color="white")
        im.paste(Image.fromarray(frame), (0, 0))
        writer.writeFrame(np.array(im))
    writer.close()


def make_webp(selected_frames, movie_fname):
    """Create a WebP file from a list of frames."""
    frame_list = [Image.fromarray(np.uint8(img), "RGB") for img in selected_frames]

    if not frame_list:
        logging.warning(f"No frames to save in {movie_fname}")
        return

    frame_list[0].save(
        movie_fname, 'WEBP', quality=50, lossless=False, save_all=True, append_images=frame_list[1:], duration=16, loop=0
    )


def replay_clip_for_savestate_and_ramdump(
    start_idx, end_idx, bk2_fpath, savestate_fname=None, ramdump_fname=None,
    skip_first_step=True, game=None, scenario=None, inttype=retro.data.Integrations.CUSTOM_ONLY
):
    """
    Replay a bk2 file up to a specific frame range.
    
    - If savestate_fname is provided, saves the savestate at start_idx.
    - If ramdump_fname is provided, saves a .npz of the state for each frame in [start_idx, end_idx).
    """
    # We only run the replay once if at least one of savestate or ramdump is needed
    replay = replay_bk2(
        bk2_fpath, skip_first_step=skip_first_step, game=game, scenario=scenario, inttype=inttype
    )

    states_list = []
    for frame_idx, (_, _, _, _, _, state) in enumerate(replay):
        if savestate_fname and frame_idx == start_idx:
            with gzip.open(savestate_fname, "wb") as fh:
                fh.write(state)
        if ramdump_fname and (start_idx <= frame_idx < end_idx):
            states_list.append(state)
        if frame_idx >= end_idx:
            break

    if ramdump_fname:
        np.savez_compressed(ramdump_fname, states_list)


def load_scenes_info(scenes_file):
    """Load scenes information from a CSV file."""
    scenes_info = pd.read_csv(scenes_file)
    scenes_info_dict = {}
    for idx, row in scenes_info.iterrows():
        try:
            scene_id = f'w{int(row["World"])}l{int(row["Level"])}s{int(row["Scene"])}'
            scenes_info_dict[scene_id] = {
                'start': int(row['Entry point']),
                'end': int(row['Exit point']),
                'level_layout': int(row['Layout'])
            }
        except:
            continue
    return scenes_info_dict


def process_bk2_file(bk2_info, args, scenes_info_dict, DERIVATIVES_FOLDER, STIMULI_PATH):
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

        # We'll always need to detect scene boundaries, so let's replay once:
        repvars, frames_list = get_variables_from_replay(
            bk2_file,
            skip_first_step=skip_first_step,
            game=args.game_name,
            inttype=retro.data.Integrations.CUSTOM_ONLY
        )
        n_frames_total = len(frames_list)
        repvars['player_x_pos'] = [
            hi * 256 + lo for hi, lo in zip(repvars['player_x_posHi'], repvars['player_x_posLo'])
        ]

        # Check if we need visual frames (gif/mp4/webp)
        needs_visual = any(ftype in args.filetypes_to_generate for ftype in ['gif', 'mp4', 'webp'])

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
                        repvars['player_x_pos'][frame_idx] >= scene_start
                        and repvars['player_x_pos'][frame_idx - 1] < scene_start
                        and repvars['player_x_pos'][frame_idx] < scene_end
                        and repvars['level_layout'][frame_idx] == level_layout
                    ):
                        start_idx = frame_idx
                        start_found = True
                else:
                    if (
                        (repvars['player_x_pos'][frame_idx] >= scene_end
                         and repvars['player_x_pos'][frame_idx - 1] < scene_end)
                        or (repvars['lives'][frame_idx] - repvars['lives'][frame_idx - 1] < 0)
                    ):
                        end_idx = frame_idx
                        start_found = False
                        scenes_info_found.append([start_idx, end_idx])
                    elif (
                        repvars['player_x_pos'][frame_idx] >= scene_start
                        and repvars['player_x_pos'][frame_idx - 1] < scene_start
                    ):
                        start_idx = frame_idx

            for pattern in scenes_info_found:
                start_idx, end_idx = pattern
                if needs_visual:
                    selected_frames = frames_list[start_idx:end_idx]
                else:
                    selected_frames = []

                clip_code = f'{rep_order_string}{str(start_idx).zfill(7)}'
                assert len(clip_code) == 14, f"Invalid clip code: {clip_code}"

                # Construct BIDS-compliant paths
                deriv_folder = op.join(DERIVATIVES_FOLDER, args.pipeline_name)
                sub_folder = op.join(deriv_folder, f"sub-{sub}")
                ses_folder = op.join(sub_folder, f"ses-{ses}")
                beh_folder = op.join(ses_folder, 'beh')
                clips_folder = op.join(beh_folder, 'clips')
                savestates_folder = op.join(beh_folder, 'savestates')
                os.makedirs(clips_folder, exist_ok=True)
                os.makedirs(savestates_folder, exist_ok=True)

                entities = (
                    f"sub-{sub}_ses-{ses}_run-{run}_level-{repvars['level']}_"
                    f"scene-{int(current_scene.split('s')[1])}_clip-{clip_code}"
                )

                # Prepare output filenames
                gif_fname       = op.join(clips_folder,      f"{entities}_beh.gif")
                mp4_fname       = op.join(clips_folder,      f"{entities}_beh.mp4")
                webp_fname      = op.join(clips_folder,      f"{entities}_beh.webp")
                savestate_fname = op.join(savestates_folder, f"{entities}_beh.state")
                ramdump_fname   = savestate_fname.replace(".state", "_ramdump.npz")
                json_fname      = op.join(clips_folder,      f"{entities}_beh.json")

                # Check which outputs still need to be generated
                needs_savestate = ('savestate' in args.filetypes_to_generate) and not op.exists(savestate_fname)
                needs_ramdump   = ('ramdump'   in args.filetypes_to_generate) and not op.exists(ramdump_fname)
                needs_gif       = ('gif'       in args.filetypes_to_generate) and not op.exists(gif_fname)
                needs_mp4       = ('mp4'       in args.filetypes_to_generate) and not op.exists(mp4_fname)
                needs_webp      = ('webp'      in args.filetypes_to_generate) and not op.exists(webp_fname)
                needs_json      = ('json'      in args.filetypes_to_generate) and not op.exists(json_fname)

                # If nothing is needed for this clip, skip it.
                if not any([needs_savestate, needs_ramdump, needs_gif, needs_mp4, needs_webp, needs_json]):
                    logging.info(f"All requested files exist for clip code {clip_code}, skipping.")
                    processing_stats['clips_skipped'] += 1
                    continue

                try:
                    # Generate GIF
                    if needs_gif:
                        make_gif(selected_frames, gif_fname)

                    # Generate MP4
                    if needs_mp4:
                        make_mp4(selected_frames, mp4_fname)

                    # Generate WebP
                    if needs_webp:
                        make_webp(selected_frames, webp_fname)

                    # Generate savestate / ramdump
                    if needs_savestate or needs_ramdump:
                        replay_clip_for_savestate_and_ramdump(
                            start_idx=start_idx,
                            end_idx=end_idx,
                            bk2_fpath=bk2_file,
                            savestate_fname=savestate_fname if needs_savestate else None,
                            ramdump_fname=ramdump_fname     if needs_ramdump   else None,
                            skip_first_step=skip_first_step,
                            game=args.game_name
                        )

                    # Generate JSON sidecar if requested
                    if needs_json:
                        metadata = {
                            'Subject': sub,
                            'Session': ses,
                            'Run': run,
                            'Level': repvars['level'],
                            'Scene': int(current_scene.split('s')[1]),
                            'ClipCode': clip_code,
                            'StartFrame': start_idx,
                            'EndFrame': end_idx,
                            'TotalFrames': n_frames_total,
                            'bk2_filepath': bk2_file,
                            'game_name': args.game_name,
                        }
                        with open(json_fname, 'w') as json_file:
                            json.dump(metadata, json_file, indent=4)

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


def collect_bk2_files(DATA_PATH, subjects=None, sessions=None):
    """Collect all bk2 files and related information from the dataset."""
    bk2_files_info = []
    for root, _, files in sorted(os.walk(DATA_PATH)):
        # Skip undesired folders
        if "sourcedata" in root:
            continue

        for file in files:
            # Look for events.tsv that are not annotated
            if "events.tsv" in file and "annotated" not in file:
                run_events_file = op.join(root, file)
                logging.info(f"Processing events file: {file}")
                events_dataframe = pd.read_table(run_events_file)
                events_dataframe = events_dataframe[events_dataframe['trial_type'] == 'gym-retro_game']

                basename = op.basename(run_events_file)
                entities = basename.split('_')
                entities_dict = {}
                for ent in entities:
                    if '-' in ent:
                        key, value = ent.split('-', 1)
                        entities_dict[key] = value

                sub = entities_dict.get('sub')
                ses = entities_dict.get('ses')
                run = entities_dict.get('run')

                if not sub or not ses or not run:
                    logging.warning(f"Could not extract subject, session, or run from filename {basename}")
                    continue

                # Apply subject/session filters if specified
                if subjects and sub not in subjects:
                    continue
                if sessions and ses not in sessions:
                    continue

                # Gather the BK2 paths from the events file
                bk2_files = events_dataframe['stim_file'].values.tolist()
                for bk2_idx, bk2_file in enumerate(bk2_files):
                    if bk2_file != "Missing file" and not isinstance(bk2_file, float):
                        bk2_files_info.append({
                            'bk2_file': bk2_file,
                            'bk2_idx': bk2_idx,
                            'sub': sub,
                            'ses': ses,
                            'run': run
                        })

    return bk2_files_info


def main(args):
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
        args.pipeline_name = 'mario_scenes_simple'
    else:
        args.game_name = 'SuperMarioBros-Nes'
        args.pipeline_name = 'mario_scenes'

    # Determine which file types to generate.
    # If none are specified, we generate them all by default (including json).
    if not args.filetypes:
        args.filetypes_to_generate = ['savestate', 'ramdump', 'gif', 'mp4', 'webp', 'json']
    else:
        args.filetypes_to_generate = args.filetypes

    # Convert to a set for easy membership checks
    args.filetypes_to_generate = set(args.filetypes_to_generate)

    # Get datapath
    DATA_PATH = op.abspath(args.datapath)

    # Load scenes info
    SCENES_FILE = args.scenesfile
    if SCENES_FILE is None:
        SCENES_FILE = op.join(
            DATA_PATH, "code", "annotations", "scenes", "resources", "scenes_mastersheet.csv"
        )
    scenes_info_dict = load_scenes_info(SCENES_FILE)

    # Setup derivatives folder
    if args.output is None:
        DERIVATIVES_FOLDER = op.join(DATA_PATH, "derivatives")
    else:
        DERIVATIVES_FOLDER = op.abspath(args.output)
    os.makedirs(DERIVATIVES_FOLDER, exist_ok=True)

    # Integrate game
    if args.stimuli_path is None:
        STIMULI_PATH = op.abspath(op.join(DATA_PATH, "stimuli"))
    else:
        STIMULI_PATH = op.abspath(args.stimuli_path)
    logging.debug(f"Adding custom stimuli path: {STIMULI_PATH}")
    retro.data.Integrations.add_custom_path(STIMULI_PATH)
    games_list = retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY)
    logging.debug(f"Available games: {games_list}")

    logging.info(f"Game to use: {args.game_name}")
    logging.info(f"Pipeline name: {args.pipeline_name}")
    logging.info(f"Generating clips for the dataset in: {DATA_PATH}")
    logging.info(f"Taking stimuli from: {STIMULI_PATH}")
    logging.info(f"Saving derivatives in: {DERIVATIVES_FOLDER}")
    logging.info(f"Using scenes file: {SCENES_FILE}")
    logging.info(f"Requested file types: {args.filetypes_to_generate}")

    # Collect all bk2 files and related information
    bk2_files_info = collect_bk2_files(DATA_PATH, args.subjects, args.sessions)
    total_bk2_files = len(bk2_files_info)

    # Process bk2 files in parallel with progress bar
    n_jobs = args.n_jobs
    logging.info(f"Processing {total_bk2_files} bk2 files using {n_jobs} job(s)...")

    with tqdm_joblib(tqdm(desc="Processing bk2 files", total=total_bk2_files)):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_bk2_file)(bk2_info, args, scenes_info_dict, DERIVATIVES_FOLDER, STIMULI_PATH)
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
        'Name': args.pipeline_name,
        'BIDSVersion': '1.6.0',
        'GeneratedBy': [{
            'Name': 'Courtois Neuromod',
            'Version': '1.0.0',
            'CodeURL': 'https://github.com/courtois-neuromod/mario_scenes/script/clip_extractor.py'
        }],
        'SourceDatasets': [{'URL': 'n/a'}],
        'License': 'CC0',
    }
    deriv_folder = op.join(DERIVATIVES_FOLDER, args.pipeline_name)
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
        "-s",
        "--scenesfile",
        default=None,
        type=str,
        help="Path to the scenes file, a CSV file that contains info about the start and end positions to clip.",
    )
    parser.add_argument(
        "-sp",
        "--stimuli_path",
        default=None,
        type=str,
        help="Path to the stimuli folder containing the game ROMs. Defaults to <datapath>/stimuli if not specified.",
    )
    parser.add_argument(
        "--filetypes",
        nargs="+",
        choices=["savestate", "ramdump", "gif", "mp4", "webp", "json"],
        help="Which output types to generate. "
             "If none is specified, all are generated by default (savestate, ramdump, gif, mp4, webp, json).",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="If set, use the simplified game version (SuperMarioBrosSimple-Nes) "
             "and output into 'mario_scenes_simple' subfolder instead of 'mario_scenes'."
    )
    parser.add_argument(
        "-n",
        "--n_jobs",
        default=-1,
        type=int,
        help="Number of CPU cores to use for parallel processing.",
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

    args = parser.parse_args()
    main(args)

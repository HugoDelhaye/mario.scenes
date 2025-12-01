"""
Scene Clip Extraction Module

This module processes Super Mario Bros replay files (.bk2) to identify and extract
individual scene clips based on player position and scene boundaries. It supports
parallel processing and multiple output formats.

Main Functions:
    - main(): Entry point for clip extraction pipeline
    - process_bk2_file(): Process a single replay file to extract scene clips
    - cut_scene_clips(): Identify scene boundaries within a replay

Output:
    BIDS-compliant directory structure with video clips, savestates, ramdumps,
    and JSON metadata sidecars.
"""

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
from videogames.utils.replay import (
    replay_bk2,
    get_variables_from_replay,
)
from videogames.utils.video import (
    make_mp4,
    make_gif,
    make_webp,
)
from videogames.utils.metadata import create_sidecar_dict, collect_bk2_files


def prune_variables(variables, start_idx, end_idx):
    """
    Trim game state variables to a specific frame range.

    Parameters
    ----------
    variables : dict
        Dictionary of game state variables with list/array values indexed by frame.
    start_idx : int
        Starting frame index (inclusive).
    end_idx : int
        Ending frame index (exclusive).

    Returns
    -------
    dict
        New dictionary with array variables trimmed to [start_idx:end_idx].
        Scalar values are preserved unchanged.
    """
    pruned_variables = {}
    for key, value in variables.items():
        if isinstance(value, (list, np.ndarray)):
            pruned_variables[key] = value[start_idx:end_idx]
        else:
            pruned_variables[key] = value
    return pruned_variables


def merge_metadata(additional_fields, base_dict):
    """
    Merge additional metadata into base dictionary without overwriting existing keys.

    Parameters
    ----------
    additional_fields : dict
        New key-value pairs to add to base_dict.
    base_dict : dict
        Existing metadata dictionary that takes precedence.

    Returns
    -------
    dict
        Updated base_dict with non-conflicting keys from additional_fields added.
    """
    for key, value in additional_fields.items():
        if key not in base_dict:
            base_dict[key] = value
    return base_dict

def get_rep_order(ses, run, bk2_idx):
    """
    Generate a unique repetition identifier string.

    Creates a zero-padded concatenation of session, run, and bk2 index
    for unique clip identification within a dataset.

    Parameters
    ----------
    ses : int
        Session number.
    run : int
        Run number.
    bk2_idx : int
        BK2 file index within the run.

    Returns
    -------
    str
        9-character identifier: {ses:03d}{run:02d}{bk2_idx:02d}
    """
    return f"{str(ses).zfill(3)}{str(run).zfill(2)}{str(bk2_idx).zfill(2)}"

def cut_scene_clips(repetition_variables, rep_order_string, scene_bounds):
    """
    Identify frame ranges where the player traverses a specific scene.

    Detects scene entry and exit by monitoring player X position relative to
    scene boundaries. A clip ends when the player crosses the scene exit point
    or loses a life.

    Parameters
    ----------
    repetition_variables : dict
        Game state variables including player_x_posHi, player_x_posLo, lives,
        level_layout, and filename.
    rep_order_string : str
        Unique repetition identifier from get_rep_order().
    scene_bounds : dict
        Dictionary with keys:
        - 'start': int, scene entry X position
        - 'end': int, scene exit X position
        - 'level_layout': int, required level layout ID

    Returns
    -------
    dict
        Mapping of clip codes (14-char strings) to (start_frame, end_frame) tuples.
        Clip code format: {rep_order}{start_frame:07d}
    """

    bk2_file = repetition_variables["filename"]

    clips_found = {}
    repetition_variables["player_x_pos"] = [
        hi * 256 + lo
        for hi, lo in zip(
            repetition_variables["player_x_posHi"],
            repetition_variables["player_x_posLo"],
        )
    ]
    n_frames_total = len(repetition_variables["player_x_pos"])

    scene_start = scene_bounds["start"]
    scene_end = scene_bounds["end"]
    level_layout = scene_bounds["level_layout"]

    start_found = False
    for frame_idx in range(1, n_frames_total):
        if not start_found:
            if (
                repetition_variables["player_x_pos"][frame_idx] >= scene_start
                and repetition_variables["player_x_pos"][frame_idx - 1]
                < scene_start
                and repetition_variables["player_x_pos"][frame_idx] < scene_end
                and repetition_variables["level_layout"][frame_idx]
                == level_layout
            ):
                start_idx = frame_idx
                start_found = True
        else:
            if (
                repetition_variables["player_x_pos"][frame_idx] >= scene_end
                and repetition_variables["player_x_pos"][frame_idx - 1]
                < scene_end
            ) or (
                repetition_variables["lives"][frame_idx]
                - repetition_variables["lives"][frame_idx - 1]
                < 0
            ):
                end_idx = frame_idx
                start_found = False
                clip_code = f"{rep_order_string}{str(start_idx).zfill(7)}"
                clips_found[clip_code] = (start_idx, end_idx)
            elif (
                repetition_variables["player_x_pos"][frame_idx] >= scene_start
                and repetition_variables["player_x_pos"][frame_idx - 1]
                < scene_start
            ):
                start_idx = frame_idx

    return clips_found


def process_bk2_file(
    bk2_info, args, scenes_info_dict, DATA_PATH, OUTPUT_FOLDER, STIMULI_PATH
):
    """
    Process a single BK2 replay file to extract all scene clips.

    For each scene in the current level, this function:
    1. Replays the BK2 file to extract frames and game state
    2. Identifies scene traversals using cut_scene_clips()
    3. Saves requested outputs: videos, savestates, ramdumps, metadata
    4. Creates BIDS-compliant directory structure and JSON sidecars

    Parameters
    ----------
    bk2_info : dict
        Replay file metadata with keys: bk2_file, bk2_idx, sub, ses, run
    args : argparse.Namespace
        Command-line arguments specifying output options:
        - save_videos, save_states, save_ramdumps, save_variables
        - video_format, game_name, replays_path
    scenes_info_dict : dict
        Scene definitions keyed by scene_id (e.g., 'w1l1s1')
    DATA_PATH : str
        Root directory of Mario dataset
    OUTPUT_FOLDER : str
        Derivatives output directory
    STIMULI_PATH : str
        Path to game ROM files

    Returns
    -------
    tuple of (list, dict)
        - error_logs: List of error message strings
        - processing_stats: Dict with keys clips_processed, clips_skipped, errors

    Notes
    -----
    This function is designed for parallel execution via joblib.
    """
    # Add stimuli path in each child process
    retro.data.Integrations.add_custom_path(STIMULI_PATH)

    error_logs = []
    processing_stats = {
        "bk2_file": bk2_info["bk2_file"],
        "clips_processed": 0,
        "clips_skipped": 0,
        "errors": 0,
    }

    try:
        bk2_file = bk2_info["bk2_file"]
        bk2_idx = bk2_info["bk2_idx"]
        sub = bk2_info["sub"]
        ses = bk2_info["ses"]
        run = bk2_info["run"]
        skip_first_step = bk2_idx == 0

        logging.info(f"Processing bk2 file: {bk2_file}")
        rep_order_string = get_rep_order(ses, run, bk2_idx)

        # Run replay
        repetition_variables, replay_info, frames_list, replay_states = (
            get_variables_from_replay(
                op.join(DATA_PATH, bk2_file),
                skip_first_step=skip_first_step,
                game=args.game_name,
                inttype=retro.data.Integrations.CUSTOM_ONLY,
            )
        )

        curr_level = op.basename(bk2_file).split("_")[-2].split("-")[1]
        scenes_in_current_level = [
            x for x in scenes_info_dict.keys() if curr_level in x
        ]
        for current_scene in scenes_in_current_level:

            scene_bounds = {'start': scenes_info_dict[current_scene]['start'],
                            'end': scenes_info_dict[current_scene]['end'],
                            'level_layout': scenes_info_dict[current_scene]['level_layout']}

            clips_found = cut_scene_clips(
                repetition_variables,
                rep_order_string,
                scene_bounds,
            )

            for clip_code in clips_found:
                start_idx, end_idx = clips_found[clip_code]

                selected_frames = frames_list[start_idx:end_idx]

                assert len(clip_code) == 14, f"Invalid clip code: {clip_code}"

                # Construct BIDS-compliant paths
                sub_folder = op.join(OUTPUT_FOLDER, f"sub-{sub}")
                ses_folder = op.join(sub_folder, f"ses-{ses}")
                beh_folder = op.join(ses_folder, "beh")

                entities = (
                    f"sub-{sub}_ses-{ses}_run-{run}_level-{repetition_variables['level']}_"
                    f"scene-{int(current_scene.split('s')[1])}_clip-{clip_code}"
                )

                # Prepare output filenames
                gif_fname = op.join(beh_folder, "videos", f"{entities}.gif")
                mp4_fname = op.join(beh_folder, "videos", f"{entities}.mp4")
                webp_fname = op.join(beh_folder, "videos", f"{entities}.webp")
                savestate_fname = op.join(beh_folder, "savestates", f"{entities}.state")
                ramdump_fname = op.join(beh_folder, "ramdumps", f"{entities}.npz")
                json_fname = op.join(beh_folder, "infos", f"{entities}.json")
                variables_fname = op.join(beh_folder, "variables", f"{entities}.json")

                metadata = {
                    "Subject": sub,
                    "Session": ses,
                    "Run": run,
                    "Level": repetition_variables["level"],
                    "Scene": int(current_scene.split("s")[1]),
                    "ClipCode": clip_code,
                    "StartFrame": start_idx,
                    "EndFrame": end_idx,
                    "TotalFrames": len(repetition_variables['score']),
                    "Bk2File": entities,
                    "GameName": args.game_name,
                    "LevelFullName": bk2_file.split("_")[-2].split("-")[1],
                    "SceneFullName": current_scene,
                }

                scene_variables = prune_variables(
                    repetition_variables, start_idx, end_idx
                )
                scene_variables["metadata"] = entities

                scene_sidecar = create_sidecar_dict(scene_variables)

                enriched_metadata = merge_metadata(metadata, scene_sidecar)

                if args.replays_path is not None:
                    replay_path = op.join(
                        args.replays_path,
                        f"sub-{sub}",
                        f"ses-{ses}",
                        "beh",
                        "infos",
                        bk2_file.split("/")[-1].replace(".bk2", ".json"),
                    )
                    if os.path.exists(replay_path):
                        with open(replay_path, "r") as replay_file:
                            replay_sidecar = json.load(replay_file)
                            enriched_metadata = merge_metadata(
                                replay_sidecar, enriched_metadata
                            )
                    else:
                        logging.warning(f"Replay file not found: {replay_path}")
                else:
                    logging.warning(
                        f"Replay path not provided, skipping repetition sidecar."
                    )

                os.makedirs(os.path.dirname(json_fname), exist_ok=True)
                with open(json_fname, "w") as json_file:
                    json.dump(enriched_metadata, json_file, indent=4)

                # If nothing is needed for this clip, skip it.
                if not any([args.save_states, args.save_ramdumps, args.save_videos]):
                    logging.info(
                        f"All requested files exist for clip code {clip_code}, skipping."
                    )
                    processing_stats["clips_skipped"] += 1
                    continue

                try:
                    # Generate files
                    if args.save_videos:
                        os.makedirs(os.path.dirname(gif_fname), exist_ok=True)
                        if args.video_format == "gif":
                            make_gif(selected_frames, gif_fname)
                        elif args.video_format == "mp4":
                            make_mp4(selected_frames, mp4_fname)
                        elif args.video_format == "webp":
                            make_webp(selected_frames, webp_fname)

                    if args.save_states:
                        os.makedirs(os.path.dirname(savestate_fname), exist_ok=True)
                        with gzip.open(savestate_fname, "wb") as fh:
                            fh.write(replay_states[start_idx])
                    if args.save_ramdumps:
                        os.makedirs(os.path.dirname(ramdump_fname), exist_ok=True)
                        np.savez_compressed(
                            ramdump_fname, replay_states[start_idx:end_idx]
                        )
                    if args.save_variables:
                        os.makedirs(os.path.dirname(variables_fname), exist_ok=True)
                        with open(variables_fname, "w") as f:
                            json.dump(scene_variables, f)

                    processing_stats["clips_processed"] += 1

                except Exception as e:
                    error_message = f"Error processing clip {clip_code} in bk2 file {bk2_file}: {str(e)}"
                    error_logs.append(error_message)
                    processing_stats["errors"] += 1
                    continue

    except Exception as e:
        bk2_file = bk2_info.get("bk2_file", "Unknown file")
        error_message = f"Error processing bk2 file {bk2_file}: {str(e)}"
        error_logs.append(error_message)
        processing_stats["errors"] += 1
        print("Full traceback:")
        traceback.print_exc()
        print("\nError message:")
        print(e)

    return error_logs, processing_stats


def main(args):
    """
    Main entry point for scene clip extraction pipeline.

    Orchestrates the complete clip extraction workflow:
    1. Load scene definitions from mastersheet
    2. Collect all BK2 replay files matching subject/session filters
    3. Process files in parallel to extract clips
    4. Generate processing logs and BIDS dataset_description.json

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments with attributes:
        - datapath: Path to Mario dataset root
        - output: Path for derivatives output
        - stimuli: Path to game ROMs (default: {datapath}/stimuli)
        - n_jobs: Number of parallel jobs (-1 = all cores)
        - save_videos, save_states, save_ramdumps, save_variables: bool flags
        - video_format: 'mp4', 'gif', or 'webp'
        - subjects: List of subject IDs to process (None = all)
        - sessions: List of session IDs to process (None = all)
        - simple: Use SuperMarioBrosSimple-Nes if True
        - verbose: Verbosity level (0=WARNING, 1=INFO, 2=DEBUG)
        - replays_path: Path to replay metadata JSON files

    Output Structure
    ----------------
    {output}/{output_name}/
        dataset_description.json
        processing_log.txt
        sub-{sub}/
            ses-{ses}/
                beh/
                    videos/
                        sub-{sub}_ses-{ses}_run-{run}_level-{level}_scene-{scene}_clip-{code}.{ext}
                    savestates/
                        sub-{sub}_ses-{ses}_run-{run}_level-{level}_scene-{scene}_clip-{code}.state
                    ramdumps/
                        sub-{sub}_ses-{ses}_run-{run}_level-{level}_scene-{scene}_clip-{code}.npz
                    infos/
                        sub-{sub}_ses-{ses}_run-{run}_level-{level}_scene-{scene}_clip-{code}.json
                    variables/
                        sub-{sub}_ses-{ses}_run-{run}_level-{level}_scene-{scene}_clip-{code}.json

    Notes
    -----
    Progress is displayed via tqdm progress bar during parallel processing.
    All errors are logged to processing_log.txt with summary statistics.
    """
    # Get datapath
    DATA_PATH = op.abspath(args.datapath)

    # Set up logging based on verbosity level
    if args.verbose == 0:
        logging_level = logging.WARNING
    elif args.verbose == 1:
        logging_level = logging.INFO
    else:
        logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level, format="%(levelname)s: %(message)s")

    # If user provides --simple, use the simpler NES version
    # and change pipeline folder name accordingly.
    if args.simple:
        args.game_name = "SuperMarioBrosSimple-Nes"
        args.output_name = "scene_clips_simple"
    else:
        args.game_name = "SuperMarioBros-Nes"
        args.output_name = "scene_clips"

    # Load scenes
    scenes_info_dict = load_scenes_info(format="dict")

    # Setup output folder
    OUTPUT_FOLDER = op.join(op.abspath(args.output), args.output_name)
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
            delayed(process_bk2_file)(
                bk2_info, args, scenes_info_dict, DATA_PATH, OUTPUT_FOLDER, STIMULI_PATH
            )
            for bk2_info in bk2_files_info
        )

    # Initialize aggregators
    total_processing_stats = {
        "total_bk2_files": total_bk2_files,
        "total_clips_processed": 0,
        "total_clips_skipped": 0,
        "total_errors": 0,
    }
    all_error_logs = []

    # Aggregate results
    for error_logs, processing_stats in results:
        total_processing_stats["total_clips_processed"] += processing_stats.get(
            "clips_processed", 0
        )
        total_processing_stats["total_clips_skipped"] += processing_stats.get(
            "clips_skipped", 0
        )
        total_processing_stats["total_errors"] += processing_stats.get("errors", 0)
        all_error_logs.extend(error_logs)

    # Prepare data for saving: BIDS derivatives dataset_description
    dataset_description = {
        "Name": args.output_name,
        "BIDSVersion": "1.6.0",
        "GeneratedBy": [
            {
                "Name": "Courtois Neuromod",
                "Version": "1.0.0",
                "CodeURL": "https://github.com/courtois-neuromod/mario.scenes/src/mario_scenes/clips_extraction/clip_extraction.py",
            }
        ],
        "SourceDatasets": [{"URL": "https://github.com/courtois-neuromod/mario/"}],
        "License": "CC0",
    }

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(op.join(OUTPUT_FOLDER, "dataset_description.json"), "w") as f:
        json.dump(dataset_description, f, indent=4)

    # Write error logs to a file
    log_file = op.join(OUTPUT_FOLDER, "processing_log.txt")
    with open(log_file, "w") as f:
        f.write("Processing Log\n")
        f.write("=================\n")
        f.write(f"Total bk2 files: {total_processing_stats['total_bk2_files']}\n")
        f.write(
            f"Total clips processed: {total_processing_stats['total_clips_processed']}\n"
        )
        f.write(
            f"Total clips skipped: {total_processing_stats['total_clips_skipped']}\n"
        )
        f.write(f"Total errors: {total_processing_stats['total_errors']}\n")
        f.write("\nError Details:\n")
        for error in all_error_logs:
            f.write(error + "\n")

    logging.info(f"Processing complete. Log file saved to {log_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract clips from Mario dataset based on scene information."
    )
    parser.add_argument(
        "-d",
        "--datapath",
        default="sourcedata/mario",
        type=str,
        help="Data path to look for events.tsv and .bk2 files. Should be the root of the Mario dataset.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default='outputdata/',
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
        "-nj",
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
        "and output into 'mario_scenes_simple' subfolder instead of 'mario_scenes'.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (can be specified multiple times)",
    )
    parser.add_argument(
        "--subjects",
        "-sub",
        nargs="+",
        default=None,
        help="List of subjects to process (e.g., sub-01 sub-02). If not specified, all subjects are processed.",
    )
    parser.add_argument(
        "--sessions",
        "-ses",
        nargs="+",
        default=None,
        help="List of sessions to process (e.g., ses-001 ses-002). If not specified, all sessions are processed.",
    )
    parser.add_argument(
        "--video_format",
        "-vf",
        default="mp4",
        choices=["gif", "mp4", "webp"],
        help="Video format to save (default: mp4).",
    )
    parser.add_argument(
        "--replays_path",
        "-rp",
        default=None,
        type=str,
        help="Path to the replay files (bk2) to extract metadata from.",
    )

    args = parser.parse_args()

    main(args)

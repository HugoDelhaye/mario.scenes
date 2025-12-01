"""Extract scene clips from Super Mario Bros replay files."""

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
from videogames.utils.replay import replay_bk2, get_variables_from_replay
from videogames.utils.video import make_mp4, make_gif, make_webp
from videogames.utils.metadata import create_sidecar_dict, collect_bk2_files


def prune_variables(variables, start_idx, end_idx):
    """Trim game state variables to frame range [start_idx:end_idx]."""
    pruned_variables = {}
    for key, value in variables.items():
        if isinstance(value, (list, np.ndarray)):
            pruned_variables[key] = value[start_idx:end_idx]
        else:
            pruned_variables[key] = value
    return pruned_variables


def merge_metadata(additional_fields, base_dict):
    """Add fields from additional_fields to base_dict without overwriting."""
    for key, value in additional_fields.items():
        if key not in base_dict:
            base_dict[key] = value
    return base_dict


def get_rep_order(ses, run, bk2_idx):
    """Generate unique repetition ID: {ses:03d}{run:02d}{bk2_idx:02d}."""
    return f"{str(ses).zfill(3)}{str(run).zfill(2)}{str(bk2_idx).zfill(2)}"


def compute_player_x_pos(variables):
    """Compute full player X position from Hi/Lo bytes."""
    return [
        hi * 256 + lo
        for hi, lo in zip(variables["player_x_posHi"], variables["player_x_posLo"])
    ]


def is_scene_start(player_x, prev_x, scene_start, scene_end, layout, level_layout):
    """Check if frame represents scene entry."""
    return (
        player_x >= scene_start
        and prev_x < scene_start
        and player_x < scene_end
        and layout == level_layout
    )


def is_scene_end(player_x, prev_x, scene_end, lives, prev_lives):
    """Check if frame represents scene exit or death."""
    crossed_boundary = player_x >= scene_end and prev_x < scene_end
    lost_life = lives - prev_lives < 0
    return crossed_boundary or lost_life


def is_scene_restart(player_x, prev_x, scene_start):
    """Check if player re-entered scene (restart after death)."""
    return player_x >= scene_start and prev_x < scene_start

def cut_scene_clips(repetition_variables, rep_order_string, scene_bounds):
    """Find all frame ranges where player traverses the scene."""
    repetition_variables["player_x_pos"] = compute_player_x_pos(repetition_variables)
    n_frames = len(repetition_variables["player_x_pos"])

    scene_start = scene_bounds["start"]
    scene_end = scene_bounds["end"]
    level_layout = scene_bounds["level_layout"]

    clips_found = {}
    start_found = False

    for frame_idx in range(1, n_frames):
        curr_x = repetition_variables["player_x_pos"][frame_idx]
        prev_x = repetition_variables["player_x_pos"][frame_idx - 1]
        curr_layout = repetition_variables["level_layout"][frame_idx]
        curr_lives = repetition_variables["lives"][frame_idx]
        prev_lives = repetition_variables["lives"][frame_idx - 1]

        if not start_found:
            if is_scene_start(curr_x, prev_x, scene_start, scene_end, curr_layout, level_layout):
                start_idx = frame_idx
                start_found = True
        else:
            if is_scene_end(curr_x, prev_x, scene_end, curr_lives, prev_lives):
                clip_code = f"{rep_order_string}{str(start_idx).zfill(7)}"
                clips_found[clip_code] = (start_idx, frame_idx)
                start_found = False
            elif is_scene_restart(curr_x, prev_x, scene_start):
                start_idx = frame_idx

    return clips_found


def get_level_from_bk2_path(bk2_file):
    """Extract level ID (e.g., 'w1l1') from BK2 filename."""
    return op.basename(bk2_file).split("_")[-2].split("-")[1]


def filter_scenes_by_level(scenes_info_dict, level):
    """Get all scene IDs belonging to specified level."""
    return [scene_id for scene_id in scenes_info_dict.keys() if level in scene_id]


def build_bids_paths(OUTPUT_FOLDER, sub, ses):
    """Create BIDS-compliant directory structure."""
    sub_folder = op.join(OUTPUT_FOLDER, f"sub-{sub}")
    ses_folder = op.join(sub_folder, f"ses-{ses}")
    beh_folder = op.join(ses_folder, "beh")
    return beh_folder


def build_entities_string(sub, ses, run, level, scene_num, clip_code):
    """Generate BIDS entities string for filenames."""
    return (
        f"sub-{sub}_ses-{ses}_run-{run}_level-{level}_"
        f"scene-{scene_num}_clip-{clip_code}"
    )


def build_output_paths(beh_folder, entities):
    """Generate all output file paths for a clip."""
    return {
        'video': {
            'gif': op.join(beh_folder, "videos", f"{entities}.gif"),
            'mp4': op.join(beh_folder, "videos", f"{entities}.mp4"),
            'webp': op.join(beh_folder, "videos", f"{entities}.webp"),
        },
        'savestate': op.join(beh_folder, "savestates", f"{entities}.state"),
        'ramdump': op.join(beh_folder, "ramdumps", f"{entities}.npz"),
        'json': op.join(beh_folder, "infos", f"{entities}.json"),
        'variables': op.join(beh_folder, "variables", f"{entities}.json"),
    }


def build_clip_metadata(bk2_info, repetition_variables, current_scene, clip_code,
                        start_idx, end_idx, args):
    """Create metadata dictionary for clip."""
    scene_num = int(current_scene.split("s")[1])
    return {
        "Subject": bk2_info["sub"],
        "Session": bk2_info["ses"],
        "Run": bk2_info["run"],
        "Level": repetition_variables["level"],
        "Scene": scene_num,
        "ClipCode": clip_code,
        "StartFrame": start_idx,
        "EndFrame": end_idx,
        "TotalFrames": len(repetition_variables['score']),
        "GameName": args.game_name,
        "SceneFullName": current_scene,
    }


def save_video_file(frames, output_path, video_format):
    """Save frames as video in specified format."""
    os.makedirs(op.dirname(output_path), exist_ok=True)
    if video_format == "gif":
        make_gif(frames, output_path)
    elif video_format == "mp4":
        make_mp4(frames, output_path)
    elif video_format == "webp":
        make_webp(frames, output_path)


def save_savestate(replay_states, start_idx, output_path):
    """Save compressed game state."""
    os.makedirs(op.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, "wb") as fh:
        fh.write(replay_states[start_idx])


def save_ramdump(replay_states, start_idx, end_idx, output_path):
    """Save compressed RAM dumps for frame range."""
    os.makedirs(op.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, replay_states[start_idx:end_idx])


def save_json_metadata(metadata, output_path):
    """Save metadata as JSON."""
    os.makedirs(op.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)


def process_single_clip(clip_code, clips_found, frames_list, replay_states,
                       repetition_variables, current_scene, bk2_info, args,
                       beh_folder, processing_stats):
    """Process and save a single scene clip."""
    start_idx, end_idx = clips_found[clip_code]
    selected_frames = frames_list[start_idx:end_idx]

    scene_num = int(current_scene.split("s")[1])
    entities = build_entities_string(
        bk2_info["sub"], bk2_info["ses"], bk2_info["run"],
        repetition_variables["level"], scene_num, clip_code
    )

    paths = build_output_paths(beh_folder, entities)
    metadata = build_clip_metadata(
        bk2_info, repetition_variables, current_scene,
        clip_code, start_idx, end_idx, args
    )

    # Enrich with scene variables
    scene_variables = prune_variables(repetition_variables, start_idx, end_idx)
    scene_variables["metadata"] = entities
    scene_sidecar = create_sidecar_dict(scene_variables)
    enriched_metadata = merge_metadata(metadata, scene_sidecar)

    # Load replay metadata if available
    if args.replays_path:
        replay_path = op.join(
            args.replays_path, f"sub-{bk2_info['sub']}", f"ses-{bk2_info['ses']}",
            "beh", "infos", bk2_info["bk2_file"].split("/")[-1].replace(".bk2", ".json")
        )
        if os.path.exists(replay_path):
            with open(replay_path, "r") as f:
                replay_sidecar = json.load(f)
                enriched_metadata = merge_metadata(replay_sidecar, enriched_metadata)

    save_json_metadata(enriched_metadata, paths['json'])

    # Save requested outputs
    if args.save_videos:
        video_path = paths['video'][args.video_format]
        save_video_file(selected_frames, video_path, args.video_format)

    if args.save_states:
        save_savestate(replay_states, start_idx, paths['savestate'])

    if args.save_ramdumps:
        save_ramdump(replay_states, start_idx, end_idx, paths['ramdump'])

    if args.save_variables:
        save_json_metadata(scene_variables, paths['variables'])

    processing_stats["clips_processed"] += 1


def process_bk2_file(bk2_info, args, scenes_info_dict, DATA_PATH, OUTPUT_FOLDER, STIMULI_PATH):
    """Process a single BK2 file to extract all scene clips."""
    retro.data.Integrations.add_custom_path(STIMULI_PATH)

    error_logs = []
    processing_stats = {"bk2_file": bk2_info["bk2_file"], "clips_processed": 0,
                       "clips_skipped": 0, "errors": 0}

    try:
        # Extract replay and get scenes for current level
        rep_order_string = get_rep_order(bk2_info["ses"], bk2_info["run"], bk2_info["bk2_idx"])
        skip_first_step = (bk2_info["bk2_idx"] == 0)

        repetition_variables, replay_info, frames_list, replay_states = (
            get_variables_from_replay(
                op.join(DATA_PATH, bk2_info["bk2_file"]),
                skip_first_step=skip_first_step,
                game=args.game_name,
                inttype=retro.data.Integrations.CUSTOM_ONLY,
            )
        )

        curr_level = get_level_from_bk2_path(bk2_info["bk2_file"])
        scenes_in_level = filter_scenes_by_level(scenes_info_dict, curr_level)
        beh_folder = build_bids_paths(OUTPUT_FOLDER, bk2_info["sub"], bk2_info["ses"])

        # Process each scene
        for current_scene in scenes_in_level:
            scene_bounds = {
                'start': scenes_info_dict[current_scene]['start'],
                'end': scenes_info_dict[current_scene]['end'],
                'level_layout': scenes_info_dict[current_scene]['level_layout']
            }

            clips_found = cut_scene_clips(repetition_variables, rep_order_string, scene_bounds)

            # Process each clip found
            for clip_code in clips_found:
                try:
                    if not any([args.save_states, args.save_ramdumps, args.save_videos]):
                        processing_stats["clips_skipped"] += 1
                        continue

                    process_single_clip(
                        clip_code, clips_found, frames_list, replay_states,
                        repetition_variables, current_scene, bk2_info, args,
                        beh_folder, processing_stats
                    )

                except Exception as e:
                    error_logs.append(f"Error processing clip {clip_code}: {str(e)}")
                    processing_stats["errors"] += 1

    except Exception as e:
        error_logs.append(f"Error processing {bk2_info.get('bk2_file', 'Unknown')}: {str(e)}")
        processing_stats["errors"] += 1
        traceback.print_exc()

    return error_logs, processing_stats


def setup_logging(verbose_level):
    """Configure logging based on verbosity level."""
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(level=levels.get(verbose_level, logging.WARNING),
                       format="%(levelname)s: %(message)s")


def configure_game_settings(args):
    """Set game name and output folder based on simple flag."""
    if args.simple:
        args.game_name = "SuperMarioBrosSimple-Nes"
        args.output_name = "scene_clips_simple"
    else:
        args.game_name = "SuperMarioBros-Nes"
        args.output_name = "scene_clips"


def setup_stimuli_path(args, DATA_PATH):
    """Configure and integrate game ROM path."""
    STIMULI_PATH = op.abspath(args.stimuli if args.stimuli else op.join(DATA_PATH, "stimuli"))
    retro.data.Integrations.add_custom_path(STIMULI_PATH)
    logging.info(f"Using game: {args.game_name}, Stimuli: {STIMULI_PATH}")
    return STIMULI_PATH


def aggregate_results(results):
    """Combine processing stats from all parallel workers."""
    total_stats = {"total_bk2_files": len(results), "total_clips_processed": 0,
                  "total_clips_skipped": 0, "total_errors": 0}
    all_errors = []

    for error_logs, stats in results:
        total_stats["total_clips_processed"] += stats.get("clips_processed", 0)
        total_stats["total_clips_skipped"] += stats.get("clips_skipped", 0)
        total_stats["total_errors"] += stats.get("errors", 0)
        all_errors.extend(error_logs)

    return total_stats, all_errors


def save_dataset_description(OUTPUT_FOLDER, output_name):
    """Create BIDS dataset_description.json."""
    description = {
        "Name": output_name,
        "BIDSVersion": "1.6.0",
        "GeneratedBy": [{"Name": "Courtois Neuromod", "Version": "1.0.0"}],
        "SourceDatasets": [{"URL": "https://github.com/courtois-neuromod/mario/"}],
        "License": "CC0",
    }
    with open(op.join(OUTPUT_FOLDER, "dataset_description.json"), "w") as f:
        json.dump(description, f, indent=4)


def save_processing_log(OUTPUT_FOLDER, total_stats, all_errors):
    """Write processing summary and errors to log file."""
    log_path = op.join(OUTPUT_FOLDER, "processing_log.txt")
    with open(log_path, "w") as f:
        f.write("Processing Log\n=================\n")
        f.write(f"Total bk2 files: {total_stats['total_bk2_files']}\n")
        f.write(f"Total clips processed: {total_stats['total_clips_processed']}\n")
        f.write(f"Total clips skipped: {total_stats['total_clips_skipped']}\n")
        f.write(f"Total errors: {total_stats['total_errors']}\n")
        f.write("\nError Details:\n")
        for error in all_errors:
            f.write(error + "\n")
    logging.info(f"Processing complete. Log: {log_path}")


def main(args):
    """Extract scene clips from BK2 replay files in parallel."""
    setup_logging(args.verbose)
    configure_game_settings(args)

    DATA_PATH = op.abspath(args.datapath)
    OUTPUT_FOLDER = op.join(op.abspath(args.output), args.output_name)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    STIMULI_PATH = setup_stimuli_path(args, DATA_PATH)
    scenes_info_dict = load_scenes_info(format="dict")
    bk2_files_info = collect_bk2_files(DATA_PATH, args.subjects, args.sessions)

    logging.info(f"Processing {len(bk2_files_info)} bk2 files with {args.n_jobs} jobs...")

    with tqdm_joblib(tqdm(desc="Processing", total=len(bk2_files_info))):
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_bk2_file)(
                bk2_info, args, scenes_info_dict, DATA_PATH, OUTPUT_FOLDER, STIMULI_PATH
            )
            for bk2_info in bk2_files_info
        )

    total_stats, all_errors = aggregate_results(results)
    save_dataset_description(OUTPUT_FOLDER, args.output_name)
    save_processing_log(OUTPUT_FOLDER, total_stats, all_errors)


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

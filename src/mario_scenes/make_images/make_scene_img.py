import retro
import numpy as np
import scipy.ndimage
import glob
import pandas as pd
from mario_replays import replay_bk2
from mario_replays.utils import get_variables_from_replay
from mario_replays.load_data import collect_bk2_files
import os.path as op
from mario_scenes.load_data import load_scenes_info
from mario_replays.load_data import load_replay_sidecars
from PIL import Image
import os
import argparse

def get_pole_position(scenes_info_dict, level_fullname='w1l1'):
    scene_end_positions = []
    for scene in scenes_info_dict.keys():
        if level_fullname in scene:
            scene_end_positions.append(scenes_info_dict[scene]['end'])
    pole_position = np.max(scene_end_positions)
    return pole_position

def average_array(arrays):
    return np.mean(np.stack(arrays), axis=0)

def main(args):
    if args.data_path is None:
        from pathlib import Path
        script_path = Path(__file__).resolve()
        repo_dir = script_path.parents[3]
        DATA_PATH = op.join(repo_dir, 'data', 'mario')
        print('Datapath not specified, using default path:', DATA_PATH)
    else:
        DATA_PATH = args.data_path
    if args.subjects == 'all':
        subjects = None
    else:
        subjects = args.subjects
    if args.simple:
        game = 'SuperMarioBrosSimple-Nes'
    else:
        game = 'SuperMarioBros-Nes'
    
    retro.data.Integrations.add_custom_path(op.join(DATA_PATH, 'stimuli'))
    scenes_info_dict = load_scenes_info(format='dict')
    bk2_list = collect_bk2_files(DATA_PATH, subjects=subjects)

    ### Determine which levels to process based on the level argument
    if args.level is not None:
        level_list = [args.level]
    else:
        level_list = np.unique([x[:4] for x in scenes_info_dict.keys()])

    chunk_size = 10

    for level_todo in level_list:
        print(f'Processing level {level_todo}')
        # First, accumulate columns across all the replays for this level
        # check if image exists
        if op.exists(op.join('..', 'resources', 'level_backgrounds', f'{level_todo}.png')):
            print(f'Background image for {level_todo} already exists. Skipping.')
            level_collected = True
        else:
            level_collected = False
        
        scenes_in_level = [x for x in scenes_info_dict.keys() if x[:4] == level_todo]
        layouts = [scenes_info_dict[x]['level_layout'] for x in scenes_in_level]
        values, counts = np.unique(layouts, return_counts=True)
        most_common_layout = values[np.argsort(counts)[-1]] # Find most common layout for level background image
        
        max_level_position = get_pole_position(scenes_info_dict, level_todo)
        
        columns_dict = {}
        for i in range(max_level_position + chunk_size):
            columns_dict[i] = []

        scenes_columns_dict = {}
        for scene in scenes_in_level:
            scenes_columns_dict[scene] = {}
            start = scenes_info_dict[scene]['start']
            end = scenes_info_dict[scene]['end']
            if end > start:
                for i in range(start, end):
                    scenes_columns_dict[scene][i] = []
            else:
                for i in range(start - 240, start):
                    scenes_columns_dict[scene][i] = []

        for bk2_info in bk2_list:
            bk2_file = bk2_info['bk2_file']
            skip_first_step = bk2_info['bk2_idx'] == 0
            bk2_level_fullname = bk2_info['bk2_file'].split('_')[-2].split('-')[1]
            if bk2_level_fullname == level_todo:
                print(f'Processing {bk2_file}')
                repetition_variables, replay_info, replay_frames, replay_states = get_variables_from_replay(
                    op.join(DATA_PATH, bk2_file),
                    skip_first_step=skip_first_step, game=game
                )
                
                # compute long version of x variables
                repetition_variables['player_x_pos'] = [
                    hi * 256 + lo for hi, lo in zip(repetition_variables['player_x_posHi'], repetition_variables['player_x_posLo'])
                ]
                repetition_variables['scroll_x_pos'] = [
                    hi * 256 + lo for hi, lo in zip(repetition_variables['xscrollHi'], repetition_variables['xscrollLo'])
                ]

                # fill dict for scenes too
                for i, frame in enumerate(replay_frames):
                    if repetition_variables['player_state'][i] == 8: # if player is dead (not active), skip frame
                        x_scroll_pos = repetition_variables['scroll_x_pos'][i]
                        x_scroll_pos_r = x_scroll_pos + 240
                        if repetition_variables['level_layout'][i] == most_common_layout:
                            if x_scroll_pos == 0:
                                for col in range(240):
                                    try:
                                        columns_dict[col].append(frame[:, col, :])
                                    except:
                                        continue
                            else:
                                for col in range(chunk_size):
                                    try:
                                        columns_dict[x_scroll_pos_r - col].append(frame[:, 240 - col, :])
                                    except:
                                        continue

                        for scene in scenes_in_level:
                            start = scenes_info_dict[scene]['start']
                            end = scenes_info_dict[scene]['end']
                            first_col = [x for x in scenes_columns_dict[scene].keys()][0]
                            #### For scenes like w1l2s12 (negative end - start) : true_start = start - 240, true_end = start
                            if x_scroll_pos_r >= start:
                                if x_scroll_pos_r <= end+240:
                                    if repetition_variables['level_layout'][i] == scenes_info_dict[scene]['level_layout']:
                                        if x_scroll_pos == 0 or repetition_variables['level_layout'][i] != most_common_layout:
                                            for col in range(240):
                                                try:
                                                    scenes_columns_dict[scene][x_scroll_pos+col].append(frame[:, col, :])
                                                except:
                                                    continue
                                        
                                        else:
                                            for col in range(chunk_size):
                                                try:
                                                    scenes_columns_dict[scene][x_scroll_pos_r - col].append(frame[:, 240 - col, :])
                                                except:
                                                    continue
                frames_shape = replay_frames[0].shape                                
                del repetition_variables, replay_frames, replay_info, replay_states

        number_of_columns = max([x for x in columns_dict.keys()])
        background_frame = np.zeros((frames_shape[0], number_of_columns, frames_shape[2]))
                
        for i in columns_dict.keys():
            try:
                column = average_array(columns_dict[i])
                background_frame[:,i,:] = column
                print(f'Column {i} done')
            except:
                print(f'Column {i} failed')
                continue
        
        img = Image.fromarray(np.uint8(background_frame))
        os.makedirs(op.join('resources', 'level_backgrounds'), exist_ok=True)
        img.save(op.join('resources', 'level_backgrounds', f'{level_todo}.png'))
        columns_dict.clear() # clear memory

        for scene in scenes_in_level:
            print(f'Processing scene {scene}')
            try:
                number_of_columns = scenes_info_dict[scene]['end'] - scenes_info_dict[scene]['start']
                background_frame = np.zeros((frames_shape[0], number_of_columns, frames_shape[2]))
                for i, column_idx in enumerate(scenes_columns_dict[scene].keys()):
                    try:
                        column = average_array(scenes_columns_dict[scene][column_idx])
                        background_frame[:,i,:] = column
                        print(f'Column {column_idx} done')
                    except:
                        print(f'Column {column_idx} failed')
                        continue

                img = Image.fromarray(np.uint8(background_frame))
                os.makedirs(op.join('resources', 'scene_backgrounds'), exist_ok=True)
                img.save(op.join('resources', 'scene_backgrounds', f'{scene}.png'))
            except:
                print(f'Failed to process scene {scene}')
                continue
        scenes_columns_dict.clear() # clear memory
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make images from bk2 files')
    parser.add_argument('-d', '--data_path', type=str, default=None, help='Path to the data directory')
    parser.add_argument('-s', '--subjects', type=str, default='sub-03', help='Subject to process')
    parser.add_argument('-l', '--level', type=str, default=None, help='Specify level to process (e.g. "w1l1"). If unspecified, all levels will be processed.')
    parser.add_argument('--simple', action='store_true', help='Use simple mode (no background image)')

    args = parser.parse_args()

    main(args)

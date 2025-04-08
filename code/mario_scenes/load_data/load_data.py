import pandas as pd
import os.path as op
import os
import json

BASE_DIR = op.dirname(op.dirname(op.dirname(op.dirname(op.abspath(__file__)))))
SCENES_MASTERSHEET = op.join(BASE_DIR, 'sourcedata', 'scenes_info' ,'scenes_mastersheet.csv')

def load_scenes_info(format='df'):
    """
    Load scenes information from a TSV file and return it in the specified format.

    Args:
        format (str): The format in which to return the scenes information. 
                      Must be either 'df' for a pandas DataFrame or 'dict' for a dictionary. 
                      Default is 'df'.

    Returns:
        pandas.DataFrame or dict: The scenes information in the specified format.

    Raises:
        ValueError: If the format is not 'df' or 'dict'.
    """
    # Check if file exists
    assert op.exists(SCENES_MASTERSHEET), f"File not found: {SCENES_MASTERSHEET}, make sure you run 'invoke collect-resources' first."
    
    # Load the data
    scenes_df = pd.read_csv(SCENES_MASTERSHEET)
    if format == 'df':
        return scenes_df
    elif format == 'dict':
        scenes_dict = {}
        for idx, row in scenes_df.iterrows():
            try:
                scene_id = f'w{int(row["World"])}l{int(row["Level"])}s{int(row["Scene"])}'
                scenes_dict[scene_id] = {
                    'start': int(row['Entry point']),
                    'end': int(row['Exit point']),
                    'level_layout': int(row['Layout'])
                }
            except:
                continue
        return scenes_dict
    else:
        raise ValueError('format must be either "df" or "dict"')
    

def load_annotation_data():
    """
    Curates the input DataFrame by creating a 'scene_ID' column and selecting specific feature columns.
    Args:
        df (pandas.DataFrame): The input DataFrame containing scene data with columns 'World', 'Level', 'Scene', and various feature columns.

    Returns:
        pandas.DataFrame: A curated DataFrame with a new 'scene_ID' column and selected feature columns.
    """
    # Create the 'scene_ID' column
    df = load_scenes_info(format='df')
    df['scene_ID'] = df.apply(
        lambda row: f"w{int(row['World'])}l{int(row['Level'])}s{int(row['Scene'])}",
        axis=1
    )
    
    # List of feature columns to keep (features and identifying variables)
    feature_cols = [
        'Enemy', '2-Horde', '3-Horde', '4-Horde', 'Roof', 'Gap',
        'Multiple gaps', 'Variable gaps', 'Gap enemy', 'Pillar gap', 'Valley',
        'Pipe valley', 'Empty valley', 'Enemy valley', 'Roof valley', '2-Path',
        '3-Path', 'Risk/Reward', 'Stair up', 'Stair down', 'Empty stair valley',
        'Enemy stair valley', 'Gap stair valley', 'Reward', 'Moving platform',
        'Flagpole', 'Beginning', 'Bonus zone'
    ]
    
    annotations_df = df[feature_cols]
    annotations_df.index = df['scene_ID']

    return annotations_df

def load_reduced_data(method='umap'):
    fname = op.join(BASE_DIR, 'outputs', 'dimensionality_reduction', f'{method}.csv')
    assert op.exists(fname), f"File not found: {fname}, make sure you run 'invoke dimensionality-reduction' first."
    return pd.read_csv(fname, index_col=0)


def load_clips_sidecars(clips_dir):
    """
    Load the sidecar files for the clips in the specified directory.

    Args:
        clips_dir (str): The directory containing the clips and their sidecar files.

    Returns:
        pandas.DataFrame: A dataframe where the rows are the individual clips and the columns are the sidecar data.
    """

    """Load sidecar files from a replay directory."""
    sidecars_data = []
    for root, folder, files in sorted(os.walk(clips_dir)):
        for file in files:
            if file.endswith(".json") and "beh" in root:
                sidecars_files = op.join(root, file)
                with open(sidecars_files) as f:
                    sidecars_data.append(json.load(f))
    sidecars_df = pd.DataFrame(sidecars_data)
    return sidecars_df
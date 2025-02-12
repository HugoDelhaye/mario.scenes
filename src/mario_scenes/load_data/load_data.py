import pandas as pd


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

    # check if file exists
    scenes_df = pd.read_csv('../resources/scenes_mastersheet.tsv', sep='\t')
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
    

def curate_dataframe(df):
    """
    Curates the input DataFrame by creating a 'scene_ID' column and selecting specific feature columns.
    Args:
        df (pandas.DataFrame): The input DataFrame containing scene data with columns 'World', 'Level', 'Scene', and various feature columns.

    Returns:
        pandas.DataFrame: A curated DataFrame with a new 'scene_ID' column and selected feature columns.
    """
    # Create the 'scene_ID' column
    df['scene_ID'] = df.apply(
        lambda row: f"w{row['World']}l{row['Level']}s{row['Scene']}",
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
    
    # Select columns to keep in the curated DataFrame
    curated_df = df[
        ['scene_ID', 'World', 'Level', 'Scene'] + feature_cols
    ].copy()
    
    return curated_df

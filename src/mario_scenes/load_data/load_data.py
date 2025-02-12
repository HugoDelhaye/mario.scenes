# TODO : function to load scenes mastersheet, return it either as a df or a dict

def load_scenes_info(format='df'):
    if format == 'df':
        scenes_df = pd.read_csv('scenes.csv')
        return scenes_df
    elif format == 'dict':
        return scenes_dict
    else:
        raise ValueError('format must be either "df" or "dict"')
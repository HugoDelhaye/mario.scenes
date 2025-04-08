# Mario Scenes Extraction
This repository contains a script to extract specific scenes from the Super Mario Bros game dataset. The script processes `.bk2` game recording files to generate video clips and savestates corresponding to predefined scenes.

This repo is a companion of the [cneuromod.mario](https://github.com/courtois-neuromod/mario.git) dataset, and is designed to be imported and reused in (non-exhaustive list) :
- [mario.annotations](https://github.com/courtois-neuromod/mario.annotations)
- [mario_learning](https://github.com/courtois-neuromod/mario_learning)
- [mario_curiosity.scene_agents](https://github.com/courtois-neuromod/mario_curiosity.scene_agents)

## Usage

- Download the repository via git : 
```
git clone git@github.com:courtois-neuromod/mario.scenes
```

### First time use
- Create an env and install the package : 
```
cd mario.scenes
python -m venv env
source env/bin/activate
pip install invoke
invoke setup-env
```

- IF ON BELUGA (or any other compute canada cluster), simply install invoke and run setup-env-on-beluga
```
pip install invoke
invoke setup-env-on-beluga
```

- Download resources (`scenes_mastersheet.tsv` in particular): 
```
invoke get-scenes-data
```

- Run analysis : 
```
invoke run-analysis
```

### To create clips
- Make sure your AWS key is exported
```
export AWS_ACCESS_KEY_ID=<s3_access_key>  AWS_SECRET_ACCESS_KEY=<s3_secret_key>
```

- Setup the mario dataset
```
invoke setup-mario-dataset
```

- Create clips (by default, creates only .json descriptive files)
```
invoke create-clips
```


## To create savestates or other files
- Run the create_clips.py script manually.
- Specify files arguments.
Example :
```
python src/mario_scenes/create_clips/create_clips.py -d data/mario --save_videos --save_states
```

## Acknowledgements

- This script uses the [Gym Retro](https://github.com/openai/retro) library for replaying game recordings.
- The BIDS standard is used for organizing the output dataset.
- This project was developed as part of the [Courtois Neuromod project](https://www.cneuromod.ca/)
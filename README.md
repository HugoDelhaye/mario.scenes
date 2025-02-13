# Mario Scenes Extraction
This repository contains a script to extract specific scenes from the Super Mario Bros game dataset. The script processes `.bk2` game recording files to generate video clips and savestates corresponding to predefined scenes.

This repo is a companion of the [cneuromod.mario](https://github.com/courtois-neuromod/mario.git) dataset, and is designed to be imported and reused in (non-exhaustive list) :
- [mario.annotations](https://github.com/courtois-neuromod/mario.annotations)
- [mario_learning](https://github.com/courtois-neuromod/mario_learning)
- [mario_curiosity.scene_agents](https://github.com/courtois-neuromod/mario_curiosity.scene_agents)

## Prerequisites
Install the [cneuromod.mario](https://github.com/courtois-neuromod/mario) dataset and the related [stimuli]https://github.com/courtois-neuromod/mario.stimuli. 
Make sure the mario dataset is on the branch `events`.

## Usage
- Download the repository via git : 
```
git clone git@github.com:courtois-neuromod/mario_scenes
```

- Create an env and install the package : 
```
mamba create -n mario_scenes
cd mario_scenes
invoke setup-env
```

- Download resources (scenes_mastersheet.tsv in particular): 
```
invoke collect-resources
```

- Run analysis : 
```
invoke run-analysis
```

## Acknowledgements

- This script uses the [Gym Retro](https://github.com/openai/retro) library for replaying game recordings.
- The BIDS standard is used for organizing the output dataset.
- This project was developed as part of the [Courtois Neuromod project](https://www.cneuromod.ca/)
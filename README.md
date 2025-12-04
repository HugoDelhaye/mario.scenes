# Mario Scenes

Extract, analyze, and cluster atomic gameplay scenes from Super Mario Bros replay data.

## Overview

This package processes Super Mario Bros gameplay recordings (.bk2 files) from the [Courtois NeuroMod project](https://www.cneuromod.ca/) to:

- **Extract** individual scene clips from full-level replays
- **Annotate** scenes with 27 gameplay features (enemies, gaps, platforms, etc.)
- **Analyze** scenes using dimensionality reduction (PCA, UMAP, t-SNE)
- **Cluster** scenes by gameplay similarity

Scenes are atomic gameplay segments with consistent mechanics (e.g., "gap with enem─ies", "staircase descent"). This decomposition enables fine-grained behavioral and neural analysis.

This package is a companion to [cneuromod.mario](https://github.com/courtois-neuromod/mario) and integrates with:

- [mario.annotations](https://github.com/courtois-neuromod/mario.annotations)
- [mario_learning](https://github.com/courtois-neuromod/mario_learning)
- [mario_curiosity.scene_agents](https://github.com/courtois-neuromod/mario_curiosity.scene_agents)

## Installation

```bash
# Clone and install
git clone git@github.com:courtois-neuromod/mario.scenes
cd mario.scenes
pip install -e .

# Download scene metadata
invoke get-scenes-data

# (Optional) Download full Mario dataset
invoke setup-mario-dataset
```

**HPC Setup (computing clusters):**

```bash
pip install invoke
invoke setup-env-on-hpc
```

## Quick Start

### 1. Extract Scene Clips

Extract video clips for each scene traversal from replay files:

```bash
# Extract clips from all replays
invoke create-clips --datapath sourcedata/mario --output outputdata/ \
  --save-videos --video-format mp4

# Process specific subjects/sessions with parallel jobs
invoke create-clips --subjects sub-01 sub-02 --sessions ses-001 --n-jobs 8
```

**Output**: BIDS-structured directories with videos, savestates, and JSON metadata:

```
outputdata/scene_clips/
└── sub-01/ses-001/beh/
    ├── videos/sub-01_ses-001_run-01_level-w1l1_scene-1_clip-*.mp4
    ├── savestates/sub-01_ses-001_run-01_level-w1l1_scene-1_clip-*.state
    └── infos/sub-01_ses-001_run-01_level-w1l1_scene-1_clip-*.json
```

### 2. Analyze Scene Features

Reduce 27-dimensional annotations to 2D for visualization:

```bash
invoke dimensionality-reduction
```

**Output**: `outputs/dimensionality_reduction/{pca,umap,tsne}.csv`

### 3. Cluster Scenes

Group scenes by gameplay similarity:

```bash
# Generate clusters with 5-30 groups
invoke cluster-scenes

# Custom cluster counts
invoke cluster-scenes --n-clusters "10 15 20"
```

**Output**: `outputs/cluster_scenes/hierarchical_clusters.pkl`

### 4. Generate Background Images

Create canonical level/scene backgrounds by averaging replay frames:

```bash
# Generate all backgrounds
invoke make-scene-images

# Specific level
invoke make-scene-images --level w1l1 --subjects sub-03
```

**Output**: `sourcedata/{level,scene}_backgrounds/*.png`

### Complete Pipeline

Run all processing steps:

```bash
invoke full-pipeline
```

## Python API

### Load Scene Data

```python
from mario_scenes.load_data import (
    load_scenes_info,
    load_annotation_data,
    load_background_images,
    load_reduced_data
)

# Load scene boundaries
scenes = load_scenes_info(format='dict')  # {scene_id: {start, end, layout}}
print(scenes['w1l1s1'])  # {'start': 0, 'end': 256, 'level_layout': 0}

# Load 27 feature annotations
features = load_annotation_data()  # DataFrame (n_scenes × 27)
print(features.loc['w1l1s1'])

# Load 2D embeddings
umap_coords = load_reduced_data(method='umap')  # DataFrame (n_scenes × 2)

# Load background images
backgrounds = load_background_images(level='scene')  # {scene_id: PIL.Image}
```

### Extract Clips Programmatically

```python
from mario_scenes.create_clips.create_clips import cut_scene_clips, get_rep_order
from videogames.utils.replay import get_variables_from_replay

# Replay a BK2 file
variables, info, frames, states = get_variables_from_replay('path/to/file.bk2')

# Find scene traversals
scene_bounds = {'start': 0, 'end': 256, 'level_layout': 0}
rep_order = get_rep_order(ses=1, run=1, bk2_idx=0)
clips = cut_scene_clips(variables, rep_order, scene_bounds)

# clips = {'0010100000000': (start_frame, end_frame), ...}
```

### Cluster Analysis

```python
from mario_scenes.scenes_analysis.cluster_scenes import generate_clusters

# Generate hierarchical clustering
clusters = generate_clusters([10, 20, 30])

# Examine 10-cluster solution
print(clusters[0]['n_clusters'])  # 10
summary = clusters[0]['summary']
print(summary[0])  # {'n_scenes': 23, 'labels': ..., 'homogeneity': ...}
```

## Scene Annotation Schema

27 binary features capture gameplay elements:

| Category      | Features                                                                       |
| ------------- | ------------------------------------------------------------------------------ |
| **Enemies**   | Enemy, 2-Horde, 3-Horde, 4-Horde, Gap enemy                                    |
| **Terrain**   | Roof, Gap, Multiple gaps, Variable gaps, Pillar gap                            |
| **Valleys**   | Valley, Pipe valley, Empty valley, Enemy valley, Roof valley                   |
| **Paths**     | 2-Path, 3-Path                                                                 |
| **Stairs**    | Stair up, Stair down, Empty stair valley, Enemy stair valley, Gap stair valley |
| **Platforms** | Moving platform                                                                |
| **Rewards**   | Risk/Reward, Reward, Bonus zone                                                |
| **Landmarks** | Flagpole, Beginning                                                            |

See `sourcedata/scenes_info/mario_scenes_manual_annotation.pdf` for details.

## Data Format

### BK2 Replay Files

Recorded with [gym-retro](https://github.com/openai/retro) at 60 Hz. Files store button presses for deterministic replay.

Expected structure:

```
sourcedata/mario/
└── sub-{subject}/ses-{session}/beh/
    ├── sub-{subject}_ses-{session}_run-{run}_level-{level}.bk2
    └── sub-{subject}_ses-{session}_run-{run}_events.tsv
```

### Output Clips

BIDS-compliant format with unique clip identifiers:

```
{output}/scene_clips/sub-{subject}/ses-{session}/beh/
├── videos/       # .mp4/.gif/.webp clips
├── savestates/   # .state files (gzipped RAM)
├── ramdumps/     # .npz files (per-frame RAM)
├── infos/        # .json metadata sidecars
└── variables/    # .json game state variables
```

Filename format: `sub-{subject}_ses-{session}_run-{run}_level-{level}_scene-{scene}_clip-{code}.{ext}`

## Available Tasks

| Task                       | Description                                         |
| -------------------------- | --------------------------------------------------- |
| `setup-env`                | Create virtual environment and install dependencies |
| `setup-env-on-hpc`         | HPC-specific environment setup                      |
| `setup-mario-dataset`      | Download Mario dataset via datalad                  |
| `get-scenes-data`          | Download scene metadata from Zenodo                 |
| `dimensionality-reduction` | Apply PCA, UMAP, t-SNE to annotations               |
| `cluster-scenes`           | Hierarchical clustering on scene features           |
| `create-clips`             | Extract scene clips from replays                    |
| `make-scene-images`        | Generate background images                          |
| `full-pipeline`            | Run complete analysis workflow                      |

Run `invoke --list` for full options.

## References

- **Dataset**: [Courtois NeuroMod](https://docs.cneuromod.ca/)
- **Scene Definitions**: [Zenodo Record 15586709](https://zenodo.org/records/15586709)
- **Related Packages**:
    - [videogames.utils](https://github.com/courtois-neuromod/videogames.utils) - Replay processing utilities
    - [airoh](https://github.com/airoh-pipeline/airoh) - Reproducible workflow framework

## Citation

If you use this package, please cite:

```bibtex
@misc{mario_scenes,
  title={Mario Scenes: Atomic Scene Decomposition for Super Mario Bros},
  author={Courtois NeuroMod Team},
  year={2025},
  url={https://github.com/courtois-neuromod/mario.scenes}
}
```

## License

MIT License - See LICENSE file for details.
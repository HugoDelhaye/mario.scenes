# mario.scenes

Extract and analyze gameplay "scenes" from Super Mario Bros recordings. A scene is a ~1 screen-width segment representing a specific design pattern or challenge.

## What Are Scenes?

Scenes divide each Mario level into meaningful gameplay segments based on player position:
- World 1-1 has 32 scenes (w1l1s01 through w1l1s32)
- Each scene = distinct challenge (first goomba, first pit, pipe sequence, etc.)
- Enables fine-grained behavioral and neural analysis

## What You Get

For each scene encountered during gameplay:
- Video clips or images of the scene
- Metadata JSON with timing, boundaries, and performance stats
- Scene similarity analysis and clustering results

## Quick Start

```bash
# Install
git clone git@github.com:courtois-neuromod/mario.scenes
cd mario.scenes
pip install -r requirements.txt
pip install -e .

# Or with airoh
pip install airoh
invoke setup-env

# Download scene definitions
invoke get-scenes-data

# Extract clips
invoke create-clips
```

## Usage

### With Airoh (Recommended)

```bash
# Extract scene clips
invoke create-clips

# With videos
invoke create-clips --save-videos --video-format mp4

# Analyze scenes
invoke dimensionality-reduction
invoke cluster-scenes

# Full pipeline
invoke full-pipeline
```

### Direct Python Script

```bash
python code/mario_scenes/create_clips/create_clips.py \
  --datapath sourcedata/mario \
  --replays_path ../mario.replays/outputdata/replays \
  --scenes_info sourcedata/scenes_info \
  --output outputdata/mario_scenes
```

## Requirements

- Python ≥ 3.8
- Mario dataset with `.bk2` files
- mario.replays outputs (game variables)
- Scene definitions (download with `invoke get-scenes-data`)

## Configuration

Edit `invoke.yaml`:

```yaml
mario_dataset: sourcedata/mario
replays_dataset: ../mario.replays/outputdata/replays
scenes_info_dir: sourcedata/scenes_info
output_dir: outputdata/mario_scenes

n_jobs: -1              # Parallel processing
save_videos: false
save_images: true
video_format: mp4       # mp4, gif, or webp
```

## Output Structure

```
outputdata/mario_scenes/
└── sub-XX/
    └── ses-XXX/
        └── clips/
            ├── scene-w1l1s01_code-XXXXX.mp4
            ├── scene-w1l1s01_code-XXXXX.json
            └── ...
```

## Available Tasks

```bash
invoke --list                    # View all tasks
invoke create-clips             # Extract scene clips
invoke get-scenes-data          # Download scene definitions
invoke dimensionality-reduction # Analyze scene features
invoke cluster-scenes           # Cluster similar scenes
invoke setup-env                # Install dependencies
```

### Task Options

**`create-clips`**:
- `--datapath` - Mario dataset location
- `--replays-path` - Path to replays outputs
- `--scenes-info` - Path to scene definitions
- `--n-jobs` - Parallel jobs (-1 = all cores)
- `--save-videos` - Generate video files
- `--video-format` - Video format (mp4, gif, webp)

**`cluster-scenes`**:
- `--n-clusters-min` - Minimum clusters (default: 5)
- `--n-clusters-max` - Maximum clusters (default: 30)

## Scene Definition Format

Scenes are defined in `scenes_mastersheet.json`:

```json
{
  "w1l1s01": {
    "start": 0,
    "end": 256,
    "scene_name": "Opening",
    "description": "First goomba encounter"
  }
}
```

Each extracted clip gets metadata:

```json
{
  "SceneFullName": "w1l1s01",
  "ClipCode": "00100100000123",
  "StartFrame": 123,
  "EndFrame": 456,
  "Duration": 5.55,
  "Cleared": true,
  "ScoreGained": 150
}
```

**Clip Code**: `{session:03d}{run:02d}{bk2_idx:02d}{start_frame:07d}` - unique sortable identifier

## Scene Applications

**Behavioral Analysis**:
- Compare performance across scene repetitions
- Analyze learning curves for specific challenges
- Identify difficulty patterns

**Neural Analysis**:
- Correlate brain activity with scene features
- Decode scene identity from fMRI
- Compare responses to similar scenes

**Computational Modeling**:
- Train RL agents on individual scenes
- Model human learning strategies

## Troubleshooting

**"Scenes info not found"**: Run `invoke get-scenes-data`

**"Replays not found"**: Process replays first with [mario.replays](https://github.com/courtois-neuromod/mario.replays)

**No clips generated**: Check that player actually entered the scene during gameplay; use `--verbose`

## Data Availability

Scene definitions and backgrounds available on [Zenodo](https://zenodo.org/records/15586709)

## Related Projects

- [mario](https://github.com/courtois-neuromod/mario) - Main dataset
- [mario.replays](https://github.com/courtois-neuromod/mario.replays) - Extract game variables (required)
- [mario.annotations](https://github.com/courtois-neuromod/mario.annotations) - Event annotations

Part of the [Courtois NeuroMod](https://www.cneuromod.ca/) project.

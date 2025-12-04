"""
Invoke tasks for mario.scenes project using airoh.

This module provides tasks for extracting and analyzing scene clips
from the Mario dataset.
"""

from invoke import task
import os
import os.path as op

# Import airoh utility tasks
from airoh.utils import setup_env_python, ensure_dir_exist, clean_folder
from airoh.datalad import get_data

BASE_DIR = op.dirname(op.abspath(__file__))


@task
def create_clips(
    c,
    datapath=None,
    replays_path=None,
    scenes_info=None,
    output=None,
    n_jobs=None,
    save_videos=False,
    save_images=True,
    video_format="mp4",
    simple=False,
    verbose=False,
):
    """
    Extract scene clips from Mario gameplay replays.

    This task processes replay files and extracts individual scene clips based on
    scene boundary definitions, generating videos/images and metadata for each clip.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context (automatically provided).
    datapath : str, optional
        Path to the mario dataset root. Defaults to mario_dataset from invoke.yaml.
    replays_path : str, optional
        Path to replays dataset. Defaults to replays_dataset from invoke.yaml.
    scenes_info : str, optional
        Path to scenes info directory. Defaults to scenes_info_dir from invoke.yaml.
    output : str, optional
        Output directory for clips. Defaults to output_dir from invoke.yaml.
    n_jobs : int, optional
        Number of parallel jobs (-1 for all cores). Defaults to n_jobs from invoke.yaml.
    save_videos : bool, optional
        Save video files for clips. Default: False.
    save_images : bool, optional
        Save image files for clips. Default: True.
    video_format : str, optional
        Video format (mp4, gif, webp). Default: mp4.
    simple : bool, optional
        Use simplified game version. Default: False.
    verbose : bool, optional
        Enable verbose output. Default: False.

    Examples
    --------
    Extract clips with default settings:
    ```bash
    invoke create-clips
    ```

    Extract with videos and verbose output:
    ```bash
    invoke create-clips --save-videos --verbose
    ```

    Use custom paths:
    ```bash
    invoke create-clips \
      --datapath /data/mario \
      --output /data/derivatives/scenes \
      --n-jobs 8
    ```
    """
    # Resolve paths from configuration or arguments
    if datapath is None:
        datapath = c.config.get("mario_dataset", "sourcedata/mario")

    if replays_path is None:
        replays_path = c.config.get("replays_dataset", "../mario.replays/outputdata/replays")

    if scenes_info is None:
        scenes_info = c.config.get("scenes_info_dir", "sourcedata/scenes_info")

    if output is None:
        output = c.config.get("output_dir", "outputdata/mario_scenes")

    if n_jobs is None:
        n_jobs = c.config.get("n_jobs", -1)

    # Validate paths
    if not op.exists(datapath):
        raise FileNotFoundError(
            f"❌ Mario dataset not found at: {datapath}\n"
            "   Run 'invoke setup-mario-dataset' or specify --datapath"
        )

    if not op.exists(scenes_info):
        raise FileNotFoundError(
            f"❌ Scenes info not found at: {scenes_info}\n"
            "   Run 'invoke get-scenes-data' to download scene definitions"
        )

    # Build command
    cmd = [
        "python",
        "code/mario_scenes/create_clips/create_clips.py",
        "--datapath", datapath,
        "--replays_path", replays_path,
        "--scenes_info", scenes_info,
        "--output", output,
        "--n_jobs", str(n_jobs),
        "--video_format", video_format,
    ]

    if save_videos:
        cmd.append("--save_videos")

    if save_images:
        cmd.append("--save_images")

    if simple:
        cmd.append("--simple")

    if verbose:
        cmd.append("--verbose")

    # Display execution info
    print("🎬 Extracting Mario scene clips...")
    print(f"   Dataset: {datapath}")
    print(f"   Replays: {replays_path}")
    print(f"   Scenes info: {scenes_info}")
    print(f"   Output: {output}")
    print(f"   Save videos: {save_videos}")
    print(f"   Save images: {save_images}")
    print()

    # Run the extraction script
    c.run(" ".join(cmd), pty=True)

    print("✅ Clip extraction complete!")


@task
def dimensionality_reduction(c, input_dir=None, output_dir=None):
    """
    Perform dimensionality reduction on scene features.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context.
    input_dir : str, optional
        Input directory with scene data.
    output_dir : str, optional
        Output directory for results.

    Examples
    --------
    ```bash
    invoke dimensionality-reduction
    ```
    """
    print("📊 Running dimensionality reduction...")
    cmd = ["python", "code/mario_scenes/scenes_analysis/dimensionality_reduction.py"]

    if input_dir:
        cmd.extend(["--input", input_dir])
    if output_dir:
        cmd.extend(["--output", output_dir])

    c.run(" ".join(cmd), pty=True)
    print("✅ Dimensionality reduction complete!")


@task
def cluster_scenes(c, n_clusters_min=None, n_clusters_max=None):
    """
    Perform hierarchical clustering on scenes.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context.
    n_clusters_min : int, optional
        Minimum number of clusters. Defaults to n_clusters_min from invoke.yaml.
    n_clusters_max : int, optional
        Maximum number of clusters. Defaults to n_clusters_max from invoke.yaml.

    Examples
    --------
    ```bash
    invoke cluster-scenes
    invoke cluster-scenes --n-clusters-min 5 --n-clusters-max 20
    ```
    """
    if n_clusters_min is None:
        n_clusters_min = c.config.get("n_clusters_min", 5)

    if n_clusters_max is None:
        n_clusters_max = c.config.get("n_clusters_max", 30)

    print(f"🔬 Clustering scenes (k={n_clusters_min} to {n_clusters_max})...")

    # Generate sequence of cluster numbers
    n_clusters_range = " ".join(str(i) for i in range(n_clusters_min, n_clusters_max + 1))

    cmd = f"python code/mario_scenes/scenes_analysis/cluster_scenes.py --n_clusters {n_clusters_range}"
    c.run(cmd, pty=True)
    print("✅ Clustering complete!")


@task
def get_scenes_data(c, output_dir=None):
    """
    Download scene definitions and background images from Zenodo.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context.
    output_dir : str, optional
        Output directory for scenes data. Defaults to scenes_info_dir from invoke.yaml.

    Examples
    --------
    ```bash
    invoke get-scenes-data
    ```
    """
    if output_dir is None:
        output_dir = c.config.get("scenes_info_dir", "sourcedata/scenes_info")

    print("📥 Downloading scene data from Zenodo...")

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("sourcedata", exist_ok=True)

    # Get URLs from config
    scenes_pdf = c.config.get("scenes_pdf_url")
    scenes_json = c.config.get("scenes_json_url")
    scenes_csv = c.config.get("scenes_csv_url")
    scene_bg = c.config.get("scene_backgrounds_url")
    level_bg = c.config.get("level_backgrounds_url")

    # Download files
    print("   Downloading scenes mastersheet (PDF)...")
    c.run(f'wget "{scenes_pdf}" -O {output_dir}/mario_scenes_manual_annotation.pdf', hide="out")

    print("   Downloading scenes mastersheet (JSON)...")
    c.run(f'wget "{scenes_json}" -O {output_dir}/scenes_mastersheet.json', hide="out")

    print("   Downloading scenes mastersheet (CSV)...")
    c.run(f'wget "{scenes_csv}" -O {output_dir}/scenes_mastersheet.csv', hide="out")

    print("   Downloading scene backgrounds...")
    c.run(f'wget "{scene_bg}" -O sourcedata/scene_backgrounds.tar.gz', hide="out")
    c.run("tar -xzf sourcedata/scene_backgrounds.tar.gz -C sourcedata/")
    os.remove("sourcedata/scene_backgrounds.tar.gz")

    print("   Downloading level backgrounds...")
    c.run(f'wget "{level_bg}" -O sourcedata/level_backgrounds.tar.gz', hide="out")
    c.run("tar -xzf sourcedata/level_backgrounds.tar.gz -C sourcedata/")
    os.remove("sourcedata/level_backgrounds.tar.gz")

    print("✅ Scene data download complete!")


@task
def setup_mario_dataset(c, use_datalad=True):
    """
    Set up the Mario dataset with replay files and stimuli.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context.
    use_datalad : bool, optional
        Use datalad to install the dataset. Default: True.

    Examples
    --------
    ```bash
    invoke setup-mario-dataset
    ```
    """
    if use_datalad:
        print("📦 Setting up Mario dataset with Datalad...")
        command = (
            "mkdir -p sourcedata && "
            "cd sourcedata && "
            "datalad install git@github.com:courtois-neuromod/mario && "
            "cd mario && "
            "datalad get */*/*/*.bk2 && "
            "datalad get */*/*/*.tsv && "
            "rm -rf stimuli && "
            "datalad install git@github.com:courtois-neuromod/mario.stimuli && "
            "mv mario.stimuli stimuli && "
            "cd stimuli && "
            "git checkout scenes_states && "
            "datalad get ."
        )
        c.run(command, pty=True)
        print("✅ Mario dataset setup complete!")
    else:
        print("⚠️  Please manually download the Mario dataset and place it in sourcedata/mario")


@task
def setup_env(c, compute_cluster=None):
    """
    Set up the Python environment for mario.scenes.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context.
    compute_cluster : str, optional
        Name of compute cluster (e.g., "beluga") for cluster-specific setup.
        Default: None (standard setup).

    Examples
    --------
    Standard setup:
    ```bash
    invoke setup-env
    ```

    Beluga cluster setup:
    ```bash
    invoke setup-env --compute-cluster beluga
    ```
    """
    print("🐍 Setting up mario.scenes environment...")

    if compute_cluster == "beluga":
        print("📦 Setting up for Beluga compute cluster...")
        c.run("module load python/3.10")
        c.run("git clone https://github.com/farama-foundation/stable-retro.git", warn=True)
        c.run("pip install -e stable-retro")

    setup_env_python(c)
    c.run("pip install -e .")
    print("✅ Environment setup complete!")


@task
def full_pipeline(c):
    """
    Run the full scene analysis pipeline.

    This includes: dimensionality reduction, clustering, and clip extraction.

    Examples
    --------
    ```bash
    invoke full-pipeline
    ```
    """
    print("🚀 Running full mario.scenes pipeline...")
    dimensionality_reduction(c)
    cluster_scenes(c)
    create_clips(c)
    print("✅ Full pipeline complete!")

"""
Mario Scenes - Invoke Tasks

This module defines reproducible workflow tasks for the mario.scenes pipeline,
following airoh pipeline conventions for task organization and documentation.

Available Tasks:
    Setup & Configuration:
        - setup-env: Create virtual environment and install dependencies
        - setup-env-on-beluga: HPC-specific environment setup for Compute Canada
        - setup-mario-dataset: Download and configure Mario dataset via datalad
        - get-scenes-data: Download scene metadata and background images from Zenodo

    Analysis & Processing:
        - dimensionality-reduction: Apply PCA, UMAP, and t-SNE to scene annotations
        - cluster-scenes: Perform hierarchical clustering on scene features
        - create-clips: Extract video clips from replay files for each scene
        - make-scene-images: Generate background images for levels and scenes

    Workflows:
        - full-pipeline: Execute complete analysis workflow
"""

from invoke import task
import os.path as op

BASE_DIR = op.dirname(op.abspath(__file__))


# ===============================
# 📦 Setup & Configuration
# ===============================

@task
def setup_env(c):
    """🔧 Set up Python virtual environment and install dependencies.

    Creates a virtual environment in ./env/ and installs all required packages
    from requirements.txt, then installs the mario_scenes package in editable mode.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.

    Examples
    --------
    ```bash
    invoke setup-env
    ```

    Notes
    -----
    This task creates a new virtual environment from scratch. If you need to update
    dependencies in an existing environment, activate it manually and run pip install.
    """
    c.run(f"python -m venv {BASE_DIR}/env && "
          f"source {BASE_DIR}/env/bin/activate && "
          "pip install -r requirements.txt && "
          "pip install -e .")


@task
def setup_env_on_beluga(c):
    """🖥️ Set up environment on Compute Canada Beluga HPC cluster.

    Configures the environment on Beluga with specific Python module loading
    and stable-retro installation from source. This task is tailored for the
    Compute Canada HPC environment.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.

    Examples
    --------
    ```bash
    invoke setup-env-on-beluga
    ```

    Notes
    -----
    This task assumes you're running on Compute Canada's Beluga cluster with
    access to the module system and git repositories.
    """
    c.run("module load python/3.10 && "
          f"python -m venv {BASE_DIR}/env && "
          "cd mario_scenes_env/lib/python3.10/site-packages && "
          "git clone git@github.com:farama-foundation/stable-retro && "
          "cd ../../../.. && "
          "source ./mario_scenes_env/bin/activate && "
          "pip install -e mario_scenes_env/lib/python3.10/site-packages/stable-retro/. && "
          "pip install -r requirements_beluga.txt && "
          "pip install -e .")


@task
def setup_mario_dataset(c):
    """📥 Download and configure the Mario dataset using datalad.

    Installs the Courtois NeuroMod Mario dataset including:
    - All .bk2 replay files
    - Event timing .tsv files
    - Game ROM stimuli

    The dataset is installed into sourcedata/mario/ following BIDS conventions.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.

    Examples
    --------
    ```bash
    invoke setup-mario-dataset
    ```

    Notes
    -----
    Requires datalad to be installed and SSH access to the Courtois NeuroMod
    repositories. The dataset is large (~several GB) and may take time to download.
    """
    command = (
        f"source {BASE_DIR}/env/bin/activate && "
        "mkdir -p sourcedata && "
        "cd sourcedata && "
        "datalad install git@github.com:courtois-neuromod/mario && "
        "cd mario && "
        "git checkout events && "
        "datalad get */*/*/*.bk2 && "
        "datalad get */*/*/*.tsv && "
        "rm -rf stimuli && "
        "datalad install git@github.com:courtois-neuromod/mario.stimuli && "
        "mv mario.stimuli stimuli && "
        "cd stimuli && "
        "git checkout scenes_states && "
        "datalad get ."
    )
    c.run(command)


@task
def get_scenes_data(c):
    """📊 Download scene metadata and background images from Zenodo.

    Downloads and extracts:
    - scenes_mastersheet.csv: Scene boundary definitions and layouts
    - scenes_mastersheet.json: Same data in JSON format
    - mario_scenes_manual_annotation.pdf: Annotation documentation
    - level_backgrounds/: Background images for each level
    - scene_backgrounds/: Background images for individual scenes

    All files are saved to sourcedata/scenes_info/ and sourcedata/*_backgrounds/.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.

    Examples
    --------
    ```bash
    invoke get-scenes-data
    ```

    Notes
    -----
    This must be run before other analysis tasks that depend on scene definitions.
    Downloads from Zenodo record 15586709.
    """
    c.run("mkdir -p sourcedata/scenes_info")
    c.run('wget "https://zenodo.org/records/15586709/files/mario_scenes_manual_annotation.pdf?download=1" -O sourcedata/scenes_info/mario_scenes_manual_annotation.pdf')
    c.run('wget "https://zenodo.org/records/15586709/files/scenes_mastersheet.json?download=1" -O sourcedata/scenes_info/scenes_mastersheet.json')
    c.run('wget "https://zenodo.org/records/15586709/files/scenes_mastersheet.csv?download=1" -O sourcedata/scenes_info/scenes_mastersheet.csv')
    c.run('wget "https://zenodo.org/records/15586709/files/scene_backgrounds.tar.gz?download=1" -O sourcedata/scene_backgrounds.tar.gz')
    c.run('wget "https://zenodo.org/records/15586709/files/level_backgrounds.tar.gz?download=1" -O sourcedata/level_backgrounds.tar.gz')
    c.run("tar -xvf sourcedata/scene_backgrounds.tar.gz -C sourcedata/")
    c.run("tar -xvf sourcedata/level_backgrounds.tar.gz -C sourcedata/")
    c.run("rm sourcedata/scene_backgrounds.tar.gz")
    c.run("rm sourcedata/level_backgrounds.tar.gz")


# ===============================
# 🔬 Analysis & Processing
# ===============================

@task
def dimensionality_reduction(c):
    """📉 Apply dimensionality reduction to scene annotation features.

    Reduces the 27-dimensional scene annotation feature space to 2D using three methods:
    - PCA (Principal Component Analysis)
    - UMAP (Uniform Manifold Approximation and Projection)
    - t-SNE (t-distributed Stochastic Neighbor Embedding)

    Results are saved as CSV files in outputs/dimensionality_reduction/.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.

    Examples
    --------
    ```bash
    invoke dimensionality-reduction
    ```

    Notes
    -----
    Requires scene annotations to be available (run get-scenes-data first).
    Output files: pca.csv, umap.csv, tsne.csv
    """
    c.run(f"python {BASE_DIR}/code/mario_scenes/scenes_analysis/dimensionality_reduction.py")


@task
def cluster_scenes(c, n_clusters=None):
    """🗂️ Perform hierarchical clustering on scene features.

    Groups scenes into clusters based on their annotation features using
    hierarchical clustering with Ward linkage. Generates cluster assignments
    and summary statistics for multiple cluster counts (default: 5-30).

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.
    n_clusters : str, optional
        Space-separated list of cluster counts to generate (e.g., "5 10 15 20").
        If not provided, generates clusters for all values from 5 to 30.

    Examples
    --------
    ```bash
    invoke cluster-scenes
    invoke cluster-scenes --n-clusters "10 15 20"
    ```

    Notes
    -----
    Output saved to outputs/cluster_scenes/hierarchical_clusters.pkl
    """
    if n_clusters is None:
        cluster_arg = " ".join(str(i) for i in range(5, 31))
    else:
        cluster_arg = n_clusters
    c.run(f"python {BASE_DIR}/code/mario_scenes/scenes_analysis/cluster_scenes.py --n_clusters {cluster_arg}")


@task
def create_clips(c, datapath="sourcedata/mario", output="outputdata/",
                 subjects=None, sessions=None, n_jobs=-1, save_videos=True,
                 video_format="mp4", simple=False):
    """🎬 Extract scene clips from Mario replay files.

    Processes .bk2 replay files to identify and extract individual scene clips,
    saving them as video files, savestates, or ramdumps. Uses parallel processing
    for efficiency.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.
    datapath : str, optional
        Path to Mario dataset root directory. Default: "sourcedata/mario"
    output : str, optional
        Path for output derivatives. Default: "outputdata/"
    subjects : str, optional
        Space-separated subject IDs to process (e.g., "sub-01 sub-02").
        If None, processes all subjects.
    sessions : str, optional
        Space-separated session IDs to process (e.g., "ses-001 ses-002").
        If None, processes all sessions.
    n_jobs : int, optional
        Number of parallel jobs. Default: -1 (use all cores)
    save_videos : bool, optional
        Whether to save video files. Default: True
    video_format : str, optional
        Video format to save: "mp4", "gif", or "webp". Default: "mp4"
    simple : bool, optional
        Use simplified game version. Default: False

    Examples
    --------
    ```bash
    invoke create-clips
    invoke create-clips --subjects "sub-01 sub-02" --video-format gif
    invoke create-clips --n-jobs 4 --simple
    ```

    Notes
    -----
    Output follows BIDS structure: sub-{sub}/ses-{ses}/beh/
    Generates JSON sidecars with metadata for each clip.
    """
    cmd = f"python {BASE_DIR}/code/mario_scenes/create_clips/create_clips.py -d {datapath} -o {output} -nj {n_jobs}"
    if save_videos:
        cmd += " --save_videos"
    cmd += f" --video_format {video_format}"
    if subjects:
        cmd += f" --subjects {subjects}"
    if sessions:
        cmd += f" --sessions {sessions}"
    if simple:
        cmd += " --simple"
    c.run(cmd)


@task
def make_scene_images(c, data_path=None, subjects="all", level=None, simple=False):
    """🖼️ Generate background images for levels and scenes.

    Processes replay files to create canonical background images by averaging
    pixel columns across multiple playthroughs. Generates both full level
    backgrounds and individual scene backgrounds.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.
    data_path : str, optional
        Path to Mario dataset. If None, uses sourcedata/mario.
    subjects : str, optional
        Subjects to include in averaging. Default: "all"
    level : str, optional
        Specific level to process (e.g., "w1l1"). If None, processes all levels.
    simple : bool, optional
        Use simplified game version. Default: False

    Examples
    --------
    ```bash
    invoke make-scene-images
    invoke make-scene-images --level w1l1 --subjects sub-03
    invoke make-scene-images --simple
    ```

    Notes
    -----
    Outputs saved to sourcedata/level_backgrounds/ and sourcedata/scene_backgrounds/
    """
    cmd = f"python {BASE_DIR}/code/mario_scenes/make_images/make_scene_img.py"
    if data_path:
        cmd += f" -d {data_path}"
    if subjects != "all":
        cmd += f" -s {subjects}"
    if level:
        cmd += f" -l {level}"
    if simple:
        cmd += " --simple"
    c.run(cmd)


# ===============================
# 🔄 Workflows
# ===============================

@task
def full_pipeline(c):
    """🚀 Execute the complete mario.scenes analysis pipeline.

    Runs all processing steps in sequence:
    1. Download scene data and metadata (get-scenes-data)
    2. Apply dimensionality reduction to features (dimensionality-reduction)
    3. Generate hierarchical clusters (cluster-scenes)
    4. Extract scene clips from replays (create-clips)

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.

    Examples
    --------
    ```bash
    invoke full-pipeline
    ```

    Notes
    -----
    Assumes the Mario dataset is already downloaded (setup-mario-dataset).
    This workflow may take several hours depending on dataset size and hardware.
    """
    c.run("invoke get-scenes-data dimensionality-reduction cluster-scenes create-clips")
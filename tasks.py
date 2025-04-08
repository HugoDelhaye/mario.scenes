from invoke import task
import os.path as op

BASE_DIR = op.dirname(op.abspath(__file__))
# ===============================
# 🔹 TASKS: Data Processing
# ===============================

@task
def dimensionality_reduction(c):
    """Cleans the raw data and saves the cleaned version."""
    c.run(f"python {BASE_DIR}/code/mario_scenes/scenes_analysis/dimensionality_reduction.py")

@task
def cluster_scenes(c):
    """Hierarchical clustering onrun_analysis scenes based on annotations."""
    c.run(f"python {BASE_DIR}/code/mario_scenes/scenes_analysis/cluster_scenes.py --n_clusters $(seq 5 30)")


# ===============================
# 🔹 TASKS: Utility & Maintenance
# ===============================

@task
def setup_env(c):
    """Sets up the virtual environment and installs dependencies."""
    c.run(f"python -m venv {BASE_DIR}/env && "
          f"source {BASE_DIR}/env/bin/activate && "
          "pip install -r requirements.txt && "
          "pip install -e .")

@task
def setup_env_on_beluga(c):
    """Sets up the virtual environment and installs dependencies on Beluga."""
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
    """Sets up the Mario dataset."""
    command = (
        f"source {BASE_DIR}/env/bin/activate && "
        "cd sourcedata && "
        "datalad install -s ria+ssh://elm.criugm.qc.ca/data/neuromod/ria-sequoia#~cneuromod.mario.raw@events mario && " #"datalad install git@github.com:courtois-neuromod/mario && "# get stimuli through submodule
        "cd mario && "
        "git checkout events && "
        "datalad get */*/*/*.bk2 && "
        "datalad get */*/*/*.tsv &&"
        "rm -rf stimuli && "
        "datalad install git@github.com:courtois-neuromod/mario.stimuli stimuli && "
        "cd stimuli && "
        "git checkout scenes_states && "
        "datalad get ."
    )
    c.run(command)

# ===============================
# 🔹 TASKS: Main
# ===============================

@task 
def create_clips(c):
    """Extracts clips from the raw data."""
    c.run(f"python {BASE_DIR}/src/mario_scenes/create_clips/create_clips.py -d data/mario")

@task
def get_assets(c):
    """Downloads and setup assets for the project."""
    c.run("mkdir -p sourcedata/scenes_info")
    c.run('wget "https://zenodo.org/records/15110657/files/mario_scenes_manual_annotation.pdf?download=1" -O sourcedata/scenes_info/mario_scenes_manual_annotation.pdf')
    c.run('wget "https://zenodo.org/records/15110657/files/scenes_mastersheet.json?download=1" -O sourcedata/scenes_info/scenes_mastersheet.json')
    c.run('wget "https://zenodo.org/records/15110657/files/scenes_mastersheet.csv?download=1" -O sourcedata/scenes_info/scenes_mastersheet.csv')
    c.run('wget "https://zenodo.org/records/15110657/files/scene_backgrounds.tar.gz?download=1" -O sourcedata/scene_backgrounds.tar.gz')
    c.run('wget "https://zenodo.org/records/15110657/files/level_backgrounds.tar.gz?download=1" -O sourcedata/level_backgrounds.tar.gz')
    c.run("tar -xvf sourcedata/scene_backgrounds.tar.gz -C sourcedata/")
    c.run("tar -xvf sourcedata/level_backgrounds.tar.gz -C sourcedata/")
    c.run("rm sourcedata/scene_backgrounds.tar.gz")
    c.run("rm sourcedata/level_backgrounds.tar.gz")

@task
def full_pipeline(c):
    """Runs the full pipeline from data cleaning to dashboard."""
    #invoke_pipeline = "invoke setup_env collect_resources dimensionality_reduction cluster_scenes"
    #c.run(f"datalad run -m 'Full pipeline execution' --python \"{invoke_pipeline}\"")
    c.run("invoke collect-resources dimensionality-reduction cluster-scenes create-clips")
from invoke import task
import os.path as op

BASE_DIR = op.dirname(op.abspath(__file__))
# ===============================
# 🔹 TASKS: Data Processing
# ===============================

@task
def run_analysis(c):
    """Cleans the raw data and saves the cleaned version."""
    c.run(f"python {BASE_DIR}/src/mario_scenes/scenes_analysis/dimensionality_reduction.py")

# ===============================
# 🔹 TASKS: Model Training
# ===============================

@task
def cluster_scenes(c):
    """Hierarchical clustering onrun_analysis scenes based on annotations."""
    c.run(f"python src/cluster_scenes.py outputs/clusters.pkl --n_clusters $(seq 5 30)")


# ===============================
# 🔹 TASKS: Utility & Maintenance
# ===============================

@task
def setup_env(c):
    """Sets up the virtual environment and installs dependencies."""
    c.run("pip install -r requirements.txt")
    c.run("pip install -e .")

@task
def clean_outputs(c):
    """Removes temporary files and cached data."""
    c.run("rm -rf outputs/*")

@task
def collect_resources(c):
    """Collects resources for the project."""
    c.run("mkdir -p resources")
    c.run('wget "https://zenodo.org/records/14847566/files/mario_scenes_manual_annotation.pdf?download=1" -O resources/mario_scenes_manual_annotation.pdf')
    c.run('wget "https://zenodo.org/records/14847566/files/scenes_mastersheet.json?download=1" -O resources/scenes_mastersheet.json')
    c.run('wget "https://zenodo.org/records/14847566/files/scenes_mastersheet.tsv?download=1" -O resources/scenes_mastersheet.tsv')

@task
def full_pipeline(c):
    """Runs the full pipeline from data cleaning to dashboard."""
    invoke_pipeline = "invoke clean_data preprocess_data train_model dashboard"
    c.run(f"datalad run -m 'Full pipeline execution' --python \"{invoke_pipeline}\"")
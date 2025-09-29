"""
Saves conda_envs.csv in conda home
"""

import sys
import os
import os.path as osp
import subprocess as sp
import pandas as pd
import json

# List of packages you want to check
packages_to_check = ["python", "numpy", "torch", "ffmpeg", "opencv-python", "diffusers", "transformers",  "huggingface_hub", "jax",  "flax", "optax"]


# Aliases for some packages
aliases = {
    "python": "py",
    "numpy": "np",
    "huggingface_hub": "hug_hub",
    "opencv-python": "cv2",
    "transformers": "transf",
}

CONDA_DIR = os.getenv("CONDA_PREFIX") if os.getenv("CONDA_DEFAULT_ENV") == "base" else osp.abspath(osp.join(os.getenv("CONDA_PREFIX"), "..", ".."))
CSV_FILE = osp.join(CONDA_DIR, "conda_envs.csv")

def get_conda_envs():
    """Return list of (env_name, env_path)."""
    result = sp.run(["mamba", "env", "list"], capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    envs = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            path = parts[2] if parts[1] == "*" else parts[1]
            envs.append((parts[0], path))
    return envs

def get_current_env():
    return os.getenv('CONDA_DEFAULT_ENV')

def get_python_version(env_name):
    """Return Python version for this environment."""
    try:
        cmd = ["mamba", "run", "-n", env_name, "python", "--version"]
        result = sp.run(cmd, capture_output=True, text=True)
        return result.stdout.strip().split()[-1]
    except Exception:
        return None


def get_mamba_packages(env_name):
    """Return {package: version} for all installed packages in env."""
    pkgs = {}
    try:
        cmd = ["conda", "run", "-n", env_name, "mamba", "list", "--export"]
        result = sp.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        for line in lines:
            pkg = line.split("=")
            if len(pkg)>=2:
                pkgs[pkg[0]] = pkg[1]
        return pkgs
    except Exception:
        return {}

def get_pip_packages(env_name):
    """Return {package: version} for all installed packages in env."""
    try:
        cmd = ["mamba", "run", "-n", env_name, "pip", "list", "--format=json"]
        result = sp.run(cmd, capture_output=True, text=True)
        pkgs = {p["name"]: p["version"] for p in json.loads(result.stdout)}
        return pkgs
    except Exception:
        return {}
def get_all_packages(env_name):
    out = get_pip_packages(env_name)
    out.update(get_mamba_packages(env_name))
    return out

def load_csv():
    if osp.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        known_envs = set(df["env"])
    else:
        df = pd.DataFrame()
        known_envs = set()
    return df, known_envs

# def add_projects(env_name):
#     if osp.exists(CSV_FILE):
#         df = pd.read_csv(CSV_FILE)
    
    # abjindex = np.where(x.values=='abj')[0].item()
    # projects = ['-', 'photoguard_2022', '-', '-', '-', '-', 'unfuss', 'sam2', 'sapiens', '-']

        # new_city_values = ['New York', 'Los Angeles', 'Chicago', 'San Francisco']

    # Step 3: Add the new column to the DataFrame
    # df['City'] = new_city_values


def main(rebuild=False):
    if rebuild:
        os.remove(CSV_FILE)
    df_existing, known_envs = load_csv()

    envs = get_conda_envs()
    rows = []
    if envs:
        print(f"known evns {known_envs}")
    i = 0
    for env_name, env_path in envs:
        if env_name in known_envs:
            i += 1
            continue
        print(f"env {env_name} -> {i}/{len(envs)}"); i += 1
        all_pkgs = get_all_packages(env_name)
        if 'huggingface-hub' in all_pkgs:
            all_pkgs['huggingface_hub'] = all_pkgs.pop('huggingface-hub')
        python_ver = get_python_version(env_name)

        row = {"env": env_name, "env_path": env_path, "python": python_ver}
        for pkg in packages_to_check:
            row[pkg] = all_pkgs.get(pkg)
        rows.append(row)

    df_new = pd.DataFrame(rows)
    df_new = df_new.rename(columns=aliases)

    if not df_existing.empty:
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df = df.fillna("-")
    df.to_csv(CSV_FILE, index=False)
    # pd.set_option("display.max_columns", None)
    # df.to_html(CSV_FILE.replace(".csv", ".html"), index=False, render_links=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        main(rebuild=True)
    else:
        main()
    print ("run streamlit run viewenvs.py # REQUIRES streamlit")

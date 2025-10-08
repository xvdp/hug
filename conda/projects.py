"""
>>> from projects import add_project
>>> project = "/home/z/work24/gits/Video/Wan2.2"
>>> add_project(project, conda='abm', force=True)
"""
import subprocess as sp
import os
import os.path as osp
from pathlib import Path
import re
import pandas as pd
import logging
import shutil

logging.basicConfig(level=logging.INFO)


def get_proj_csv(name="projects.csv"):
    conda_dir = os.getenv("CONDA_PREFIX")
    assert isinstance(conda_dir, str) and osp.isdir(conda_dir)
    if os.getenv("CONDA_DEFAULT_ENV") != "base":
        conda_dir = osp.abspath(osp.join(conda_dir, "..", ".."))
    return osp.join(conda_dir, name)



def _add_project(project, conda=None, docker=None, force=False, **kwargs):

    fname = get_proj_csv()

    # Load or initialize CSV
    if os.path.exists(fname):
        df = pd.read_csv(fname, dtype=str).fillna('-')
    else:
        df = pd.DataFrame(columns=["project", "conda", "docker"])
    
    # Either conda/docker is provided if project doesn't exist
    if conda is None and docker is None and project not in df["project"].values:
        raise ValueError("At least one of 'conda' or 'docker' must be provided.")

    # Make sure all necessary columns exist
    for col in ["project", "conda", "docker"] + list(kwargs.keys()):
        if col not in df.columns:
            df[col] = "-"


    if project in df["project"].values:
        row_idx = df.index[df["project"] == project][0]

        # Update or check each column
        for col, val in [("conda", conda), ("docker", docker), *kwargs.items()]:
            if val is None:
                continue  # skip if nothing provided
            current_val = df.at[row_idx, col] if col in df.columns else "-"
            if current_val != "-" and current_val != val:
                if not force:
                    logging.error(
                        f"Conflict for project '{project}', column '{col}': "
                        f"existing='{current_val}', new='{val}'. Use force=True to overwrite."
                    )
                    raise ValueError("Conflict in project update. See logs.")
                else:
                    logging.warning(
                        f"Overwriting project '{project}', column '{col}': "
                        f"'{current_val}' -> '{val}'"
                    )
            df.at[row_idx, col] = val
    else:
        # Fill all values with "-" initially
        new_row = {col: "-" for col in df.columns}
        new_row["project"] = project
        if conda is not None:
            new_row["conda"] = conda
        if docker is not None:
            new_row["docker"] = docker
        for k, v in kwargs.items():
            new_row[k] = v if v is not None else "-"
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save back
    df.to_csv(fname, index=False)


def delete_project(project):
    
    fname = get_proj_csv()
    if not os.path.exists(fname):
        logging.warning("CSV file does not exist.")
        return
    df = pd.read_csv(fname, dtype=str).fillna('-')
    if project not in df["project"].values:
        logging.warning(f"Project '{project}' not found in CSV.")
        return
    df = df[df["project"] != project]
    df.to_csv(fname, index=False)
    logging.info(f"Deleted project '{project}'.")


def add_cell(project, column, value, write: bool = True):
    """
    """
    fname = get_proj_csv()
    if not os.path.exists(fname):
        raise FileNotFoundError(f"{fname} file does not exist, run add_project instead")
    
    df = pd.read_csv(fname, dtype=str).fillna('-')
    if project not in df["project"].values:
        logging.warning(f"Project '{project}' not found in CSV, run add_project instead")
        return df
    
    if column not in df:
        df.insert(len(df.columns) - 1, column, "-")
        
    idx = df.index[df["project"] == project][0]
    df.at[idx, column] = value
    if write:
        shutil.copyfile(fname, fname+".bak")
        df.to_csv(fname, index=False)
    return df


def col_set(column, place: int = -1, write: bool = True):
    """ add e column with name at place
    """
    fname = get_proj_csv()
    if not os.path.exists(fname):
        raise FileNotFoundError(f"{fname} file does not exist.")

    shutil.copyfile(fname, fname+".bak")
    df = pd.read_csv(fname, dtype=str).fillna('-')
    # Fixed columns
    fixed = ["project", "conda", "docker"]
    missing = set(fixed) - set(df.columns)
    if missing:
        raise ValueError(f"MColumns '{missing}' not in {fname}.")
    # other
    keys = [c for c in df.columns if c not in fixed]
    if column not in keys:
        raise ValueError(f"Col '{column}' not in {fname}.")
    keys.remove(column)
    numcols = len(keys) + 1 
    place = min(place, len(keys))% numcols
    keys.insert(place, column)
    df = df[fixed + keys]
    if write:
        shutil.copyfile(fname, fname+".bak")
        df.to_csv(fname, index=False)
    return df


def get_conda_python_version(conda):
    try:
        cmd = ['python', '--version']
        if conda is not None and conda != '-':
            cmd = ['conda', 'run', '-n', conda] + cmd
        result = sp.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip().split()[-1]
    except sp.CalledProcessError as e:
        print(f"Error: {e}")
        return None
           
def _get_active_conda_env():
    """Return name of active conda env if available."""
    return os.environ.get("CONDA_DEFAULT_ENV", "-")

def _get_installed_packages(conda):
    """Return dict of installed packages -> version from conda or pip."""
    pkgs = {'python': get_conda_python_version(conda)}
    try:
        if conda and conda != "-":
            cmd = ["conda", "list", "-n", conda]
        else:
            cmd = ["conda", "list"]
        result = sp.run(cmd, capture_output=True, text=True, check=True)

        _cuda = False
        for line in result.stdout.splitlines():
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    pkg  =parts[0].lower()
                    version = parts[1]
                    pkgs[pkg] = version
                    if pkg in ("cudatoolkit", "cuda-toolkit") and not _cuda:
                        pkgs["cuda"] = version
                        _cuda = True
                    if pkg in ("cuda", "cuda-version"):
                        pkgs["cuda"] = version
                        _cuda = True
        return pkgs
    except Exception:
        # fallback to pip
        result = sp.run(["pip", "list", "--format=freeze"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "==" in line:
                name, ver = line.split("==", 1)
                pkgs[name.lower()] = ver
        return pkgs

def _parse_requirements(path, filterreq=None):
    """Return dict {package: constraint} for pinned packages from requirements files."""
    pinned = {}
    req_re = re.compile(r"^([A-Za-z0-9_\-]+)(\[.*\])?([<>=!~].+)$")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = req_re.match(line)
            if m:
                pkg = m.group(1).lower()
                if filterreq is None or pkg in filterreq:
                    constraint = m.group(3)
                    pinned[pkg] = constraint
    return pinned


def _resolve_project(project=None):
    if project is None:
        cwd = Path.cwd()
        project = cwd.name
        path = str(cwd.resolve())
    else:
        path = None
        p = Path(project)
        if p.exists() and p.is_dir():
            path = str(p.resolve())
            project = p.name
    return project, path


def check_cuda(version):
    pass

def add_project(project=None, conda=None, docker=None, force=False, libs=(), all_requirements=True, **kwargs):
    """
    add_project("ControlNet", conda="ab310", libs=("python", "numpy", "torch", "cuda", "torchvision", "transformers", "diffusers"), all_requirements=False, ,path="/home/z/work24/gits/Diffusion/ControlNet")
    
    TODO: akll requirements False, should check req. before list
    TODO setting path, should not check installed. 

    """
    project, path = _resolve_project(project)
    installed = None 
    required = ["project", "conda", "docker", "python", "path"]
    libs = list(libs)


    # CSV load or init
    fname = get_proj_csv()
    if os.path.exists(fname):
        df = pd.read_csv(fname, dtype=str).fillna("-")
        shutil.copyfile(fname, fname+".bak")
    else:
        df = pd.DataFrame(columns=required)
    # Ensure required columns
    for col in required:
        if col not in df.columns:
            df[col] = "-"


    # If path is set, parse requirements
    pkgs = {}
    if project in df["project"].values:
        idx = df.index[df["project"] == project][0]
        pkgs = {k:v.split(" | ")[0] for k,v in df.loc[idx].items() if v != "-" and "|" in v}

    pkgs = {**kwargs}
    if path:
        for reqfile in Path(path).glob("requirements*.txt"):
            filterreq = None if all_requirements else libs
            pkgs.update(_parse_requirements(reqfile, filterreq))

    # Build row template
    new_row = {col: "-" for col in df.columns}
    new_row["project"] = project
    if path: new_row["path"] = path

    # Handle conda detection if path exists and conda not given
    if docker is not None:
        new_row["docker"] = docker
        # in another time parse docker? less important
    else:
        conda = conda or _get_active_conda_env()
        if conda is not None:
            new_row["conda"] = conda
            installed = _get_installed_packages(conda)

    # add columns csv
    if 'python' not in pkgs.keys():
        libs.append('python')

    for pkg, constraint in pkgs.items():
        if pkg not in df.columns:
            # Insert pinned package column before 'path'
            df.insert(len(df.columns) - 1, pkg, "-")
        version = "" if installed is None or pkg=='path' else " | " + installed.get(pkg, "X")
        new_row[pkg] = f"{constraint}{version}"

    if project in df["project"].values:
        idx = df.index[df["project"] == project][0]
        exist_keys = {k for k,v in df.loc[idx].items() if v != "-" and "|" not in v}
        libs += exist_keys
    
    for pkg in libs:
        if pkg not in df.columns:
            # Insert pinned package column before 'path'
            df.insert(len(df.columns) - 1, pkg, "-")
        new_row[pkg] = installed.get(pkg, "-")

    # # Insert row or update existing
    if project in df["project"].values:
        idx = df.index[df["project"] == project][0]
        if not force:
            raise ValueError(f"Project {project} already exists. Use force=True to overwrite.")
        df.loc[idx] = pd.Series(new_row)
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Reorder so 'path' is last
    cols = [c for c in df.columns if c != "path"] + ["path"]
    df = df[cols]

    df.to_csv(fname, index=False)
    return df

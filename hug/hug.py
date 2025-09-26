from typing import Union, Optional
import huggingface_hub as hf_hub
from huggingface_hub.errors import HFValidationError, HfHubHTTPError
from huggingface_hub import hf_hub_download, scan_cache_dir
import os
import os.path as osp
import json
from pprint import pprint

def get_hf_home():
    import os
    home = os.getenv('HUGGINGFACE_HOME') or os.path.expanduser("~/.cache/huggingface")
    assert os.path.isdir(home), f" huggingface home folder does not exist in {home}"
    return home

def list_models():
    for repo in hf_hub.list_cached_models():
        print(repo.repo_id, repo.repo_type, repo.size_on_disk // 1024**2, "MB")

def get_model_files(model= "stable-diffusion-v1-5/stable-diffusion-inpainting"):
    """"""
    cache_root = hf_hub.snapshot_download(model, local_files_only=True) 

    for root, _, files in os.walk(cache_root):
        level = root.replace(cache_root, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 2 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

def param_summary(model, name):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name:15} | total {total/1e6:7.2f} M | trainable {trainable/1e6:7.2f} M")

# for n, c in pipe_inpaint.components.items():
#     if hasattr(c, 'parameters'):
#         param_summary(c, n)

def list_models(tag="diffusers", search="stabilityai/stable-diffusion"):
    from huggingface_hub import list_models
    print("\n".join([(m.modelId) for m in
                 list_models(filter="diffusers", search="stabilityai/stable-diffusion", sort="downloads", direction=-1, limit=40)]))


def _modelid_to_folder(model_id: str) -> str:
    return f"models--{model_id.replace("/", "--")}"

def _folder_to_modelid(folder: str) -> str:
    folder = osp.basename(folder)
    parts = folder.split("--")
    if folder.startswith("models--") and len(parts) >= 3:
        return f"{parts[1]}/{parts[2]}"
    return None

def _folders_to_modelids(folders: Union[list,tuple,str]) -> list:
    if isinstance(folders, str):
        folders = (folders, )
    out = ["/".join(f.replace("models--", "").split("--")[:2]) for f in osp.basename(folders) if
           (osp.isdir(f) and f.startswith("models--") and len(f.split("--")) >=3)]
    return out

def _get_snapshots(model_folder: str) -> list:
    out = []
    snapshot_dir = osp.join(model_folder, 'snapshots')
    if osp.isdir(snapshot_dir):
        out = [f.path for f in os.scandir(snapshot_dir) if f.is_dir]
    return out

def _sorted_commits(model_id: str ) -> list:
    """ requires connection"""
    api = hf_hub.HfApi()
    commits = api.list_repo_commits(model_id, repo_type="model")
    return {c.commit_id: i for i, c in
            enumerate(sorted(commits, key=lambda c: c.created_at, reverse=True))}

def _model_exists(model_id: str) -> bool:
    api = hf_hub.HfApi()
    try:
        api.model_info(model_id)
        return True
    except HFValidationError as e:
        print (f" model_id {model_id} not found", e)
    return False

def _get_model_dirs(cache_home: Union[str, list, tuple, None] = None,) -> list:
    home = cache_home or get_hf_home()
    _legacy_dirs = ['hub', 'diffusers', 'transformers']
    if isinstance(home, str):
        home = [home]
    folders = []
    for h in home:
        for d in _legacy_dirs:
            _d = osp.join(h, d)
            if os.path.isdir(_d):
                folders.append(_d)
    if cache_home is not None:
        folders = home + folders
    return folders

"""
from huggingface_hub import list_models
print("\n".join([(m.modelId) for m in
                 list_models(filter="diffusers", search="stabilityai/stable-diffusion", sort="downloads", direction=-1, limit=40)]))
print("\n".join([(m.modelId) for m in
                 list_models(filter="diffusers", search="stable-diffusion-v1-", sort="downloads", direction=-1, limit=40)]))

                 
dict_keys(['id', 'author', 'sha', 'last_modified', 'created_at', 'private', 'gated', 'disabled', 'downloads',
'downloads_all_time', 'likes', 'library_name', 'gguf', 'inference', 'inference_provider_mapping', 'tags', 'pipeline_tag',
'mask_token', 'trending_score', 'card_data', 'widget_data', 'model_index', 'config', 'transformers_info', 'siblings', 'spaces',
'safetensors', 'security_repo_status', 'xet_enabled', 'lastModified', 'cardData', 'transformersInfo', '_id', 'modelId'])

"""

def list_local_snapshots(cache_home: Union[str, list, tuple, None] = None,
                         verbose: bool=True) -> dict:
    """ lists all snapshots under cache_home /hub /transformers /diffusers
    Args
        cache_home (str, list) = None -> $HUGGINGFACE_HOME or ~/.cache/huggingface
        verbose (bool [True])  -> pprint snapshots dict

    similar to 
    $ huggingface-cli scan-cache # except this scans not just hub/ but diffusers/ ant transformers/
    """
    folders = []
    for f in _get_model_dirs(cache_home):
        folders += [f.path for f in os.scandir(f) if f.is_dir()
                    and _folder_to_modelid(f.name) is not None]
    #return folders
    out = {}
    _out = {}
    empty = {}

    for f in folders:
        k = _folder_to_modelid(f)

        if k not in _out:
            _out[k] = []
        _out[k] +=  _get_snapshots(f)

    for k, v in _out.items():
        if len(v):
            out[k] = v
        else:
            empty[k] = []
    if verbose:
        pprint(out)
    return out


def get_local_snapshots(model_id: str,
                        cache_home: Union[str, list, tuple, None] = None,
                        download: bool = False) -> list:
    """ find local shapshots and (if internet connection) orders them, ->out[0] is latest
    checks under cache_home (default $HUGGINGFACE_HOME or ~/.cache/home)
    Args
        model_id    (str)  ->  [snapshot dirs]
        cache_home (str, list) = None -> $HUGGINGFACE_HOME or ~/.cache/huggingface
        download (bool [False]) -> if True and no model found, download newest
    """

    folders = _get_model_dirs(cache_home)

    _target_folder = _modelid_to_folder(model_id)
    out = []
    for f in folders:
        out += _get_snapshots(osp.join(f, _target_folder))

    if not out:
        print(f" no local model found")
        if download:
            out = [hf_hub.snapshot_download(model_id)]
    try:
        commits = _sorted_commits(model_id)
        if len(out) > 1:
            snapshot_key=lambda path: commits.get(osp.basename(path), float('inf'))
            out = sorted(out, key=snapshot_key)
        commits = list(commits.keys())
        if osp.basename(out[0]) != commits[0]:
            print(f"there is a newer commit that locally stored at {out[0]}, see {'\n'.join(commits)}")
    except HfHubHTTPError as e:
        print(f"not ordering commits, no internet connection found, {e}")
    return out

def get_snapshot_pipeline(snapshot):
    """ From a local snapshot, get which pipleine it applies to
    """
    cf = [f.path for f in os.scandir(snapshot) if f.name=="model_index.json"][0]
    if cf:
        cfg = json.load(open(cf))
        return cfg.get("_class_name")
    else:
        print("incomplete snapshot, download with model_index.json")
    return None


def get_pipeline_snapshots(pipeline="StableDiffusionInpaintPipeline", local_files_only=False):
    """ from a pipleine with a task, list app,icable snapshots. snapshots.
    WIP TODO : fix, clean , generalize and TEST
    looks into hugging face to which 
    """

    api = hf_hub.HfApi()

    # 1. grab candidate repos from the Hub
    repos = api.list_models(
        filter="diffusers",
        task="text-to-image-inpainting",   # or "text-to-image", "image-to-image" …
        sort="downloads",
        direction=-1,
        limit=200,
    )

    # 2. keep only the ones that really use the desired pipeline
    compatible = []                       # list of (repo_id, snapshot_hash)
    for r in repos:
        try:
            cfg_path = hf_hub_download(r.id, "model_index.json", local_files_only=local_files_only)
            cls_name = json.load(open(cfg_path)).get("_class_name")
            if cls_name == pipeline:
                compatible.append(r.id)
        except Exception:
            continue                       # private / corrupted / missing file

    print(f"Repos that can be loaded with {pipeline}:")
    for rid in compatible:
        print(" -", rid)

    # 3. (optional) show which snapshots you already have locally
    print("\nLocal snapshots:")
    report = scan_cache_dir()
    for repo in report.repos:
        if repo.repo_id in compatible:
            print(" -", repo.repo_id, "→", repo.revision[:7])




import diffusers
from diffusers import StableDiffusionInpaintPipeline
import os
import os.path as osp
import shutil

from .hug import get_hf_home, get_local_snapshots

def list_pipelines(key="StableDiffusion", i=False):
    case = lambda text, i: text.lower() if i else text
    if key is not None:
        key = case(key, i)
    pipelines = diffusers._import_structure['pipelines'] 
    if key is None:
        return pipelines
    return [p for p in pipelines if key in case(p, i)]

def load_pipeline(pipeline, model_id, download=False, **kwargs):
    _pipelines = diffusers._import_structure['pipelines']
    assert pipeline in _pipelines, f"pipeline {pipeline} not found, among {_pipelines}"
    snaps = get_local_snapshots(model_id)
    if snaps:
        model_id = snaps
    elif not download:
        print(f"no local snapshot for model_id {model_id} found")
        return None
    return getattr(diffusers, pipeline).from_pretrained(model_id,  **kwargs)

def stable_diffusion_inpaint_pipeline(model_id, download=False, **kwargs):
    snaps = get_local_snapshots(model_id)
    if snaps:
        model_id = snaps
    elif not download:
        print(f"no local snapshot for model_id {model_id} found")
        return None
    return StableDiffusionInpaintPipeline.from_pretrained(model_id,  **kwargs)


# def update_cache(model="stable-diffusion-v1-5/stable-diffusion-v1-5" ):
#     """ model can be a folder or a tag
#     """
#     home = get_hf_home()
#     kw={}
#     if osp.isdir(model):
#         new_folder= osp.join(home, "hub", osp.basename(model) )
#     else:
#         new_folder= osp.join(home, "hub", f"models--{model.replace("/", "--")}" )
#         old_cache= osp.expanduser(osp.join(home, "diffusers"))
#         kw = {"cache_dir": old_cache}
#         if not osp.isdir(old_cache):
#             print(f"nothing done, old model cache {old_cache} missing")
#             return 1

#     if osp.isdir(new_folder):
#         print(f"nothing done, model exists in {new_folder}")
#         return 1
#     pipe = StableDiffusionInpaintPipeline.from_pretrained(model, **kw)
#     pipe.save_pretrained(os.path.expanduser(new_folder))
#     print(f"saving mode to {new_folder}: {osp.isdir(new_folder)}")
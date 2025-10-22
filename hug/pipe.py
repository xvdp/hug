""" generic pipeline handler

list_pipelines()    -> list of diffusers pipelines
load_pipeline()     -> generic loading from pretrained
inspect_pipeline()  -> components and parameters

"""
from typing import Optional, Any
import diffusers
import os.path as osp
import torch

from .hug import get_local_snapshots


def list_pipelines(key: Optional[str] = None, i: bool = False, w: bool = False):
    """ list pipelines available in hugging face with key
    Args
        key:    None,   list all pipelines
                str,    list pipelines matching pattern, default case sensitive, partial match
        i:      bool [False],   case insensitive match
        w:      bool [False],   match whole world
    Example:
        >>> list_pipelines("stableaudio", i=True) -> ['StableAudioPipeline', 'StableAudioProjectionModel']
    """
    case = lambda text, i: text.lower() if i else text
    if key is not None:
        key = case(key, i)
    pipelines = diffusers._import_structure['pipelines'] 
    if key is None:
        return pipelines
    return [p for p in pipelines if key in case(p, i)]


def load_pipeline(pipeline: str, model_id: str, download: bool = False, **kwargs):
    """ generic <pipeline>.from_pretrained(model_id, **kwargs)
    searches local snapshots first before downloading
    Example 
        >>> pipeline = 'StableDiffusionImg2ImgPipeline'
        >>> model_id='stable-diffusion-v1-5/stable-diffusion-v1-5'
        >>> pipe = load_pipeline(pipeline, model_id, torch_dtype=torch.float16)
    """
    _pipelines = diffusers._import_structure['pipelines']
    assert pipeline in _pipelines, f"pipeline {pipeline} not found, among {_pipelines}"
    snaps = get_local_snapshots(model_id)
    if snaps:
        model_id = snaps[0]
    elif not download:
        print(f"no local snapshot for model_id {model_id} found")
        return None
    return getattr(diffusers, pipeline).from_pretrained(model_id,  **kwargs)



def load_img2img_pipe(model_id='stable-diffusion-v1-5/stable-diffusion-v1-5'):
        pipeline = 'StableDiffusionImg2ImgPipeline'
        return load_pipeline(pipeline, model_id, torch_dtype=torch.float16)

def load_inpaint_pipe(model_id='stable-diffusion-v1-5/stable-diffusion-inpainting'):
        pipeline = 'StableDiffusionInpaintPipeline'
        return load_pipeline(pipeline, model_id, torch_dtype=torch.float16)

def inspect_pipeline(pipe) -> None:
    """
    """
    if hasattr(pipe, "__repr__"):
        print(pipe.__repr__().splitlines()[0].replace("{","").strip())
    params = {}
    utils = {}
    for k in pipe.__dict__:
        if hasattr(pipe.__dict__[k], "parameters"):
            params[k] =  count_params(pipe.__dict__[k])
        else:
            utils[k] = count_params(pipe.__dict__[k])
    totalp = 0
    for k,v in params.items():
        totalp += v
        vs = f"{v:,}"
        print(f"  {k:<14} {vs:>16}")
    tps = f"{totalp:,}"
    print(f"{'total params:':<16} {tps:>16}")
    print("utils:")
    for k,v in utils.items():
        print(f"  {k:<20} {v}")

def count_params(obj):
    out = None
    if hasattr(obj, "parameters"):
        out = sum([p.numel() for p in obj.parameters()])
    elif obj is None:
        out = ""
    else:
        out = type(obj)
    return out



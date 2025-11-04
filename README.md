# hug
## perhaps this jumble of docs and poor code should be private, iti snot clean, code or docs, it contains a mix of stuff. but I find it simpler to leave it out there than deal with passwords.
basic hugging face handling scripts. Hugging space api is not clear to me, so as i use it i document it here.
* HF requires login and tokens to download  `huggingface-cli login`
* .cn alternative to huggingface: [modelscope](https://modelscope.cn/home). 

hugging face keeps chaning. 

## Components
* model cache:    hf stores model in a hashed, and symlinked form, unless specified sored in cache home
* deployment:
    * remote - not explored here
    * local
        * some projects use hf only as model weight storage
        * pipelines -> diffusers, trransformers
* cli some info on cli commands
* conda environment info and project constraint mgmt - dependencies are a tangle mess, pip and conda require very specific constraining to not loose.g.
    * numpy 1.X -> 2. broke many things, opencv-python
    * projects using cuda compilation require specific version
    * google colab code breaks stuff (including google's own code) by going bleeding edge on python and numpy.

## Cache home 
* Local machine default `~/.cache/huggingface`, or `$HUGGINGFACE_HOME` 
* subfolders: models now got to the `<cachehome>/hub`, before to `<>/diffusers` and `<>/transformers`

## CLI cmds
* `models = list_models( filter='', search='', sort='', limit='', ... ) -> iterator` [HF Notes](HFNOTES.md) # 2025.09 -> 2M+ models
* `mod = next(models)` -> `.id` (name)==`.modelId` (name), `._id` (snapshot hash), `.pipeline_tag` (task), `.private` (bool), `.tags` (metadata list, papers, etc)

### huggingface-cli  Deprecation notice
```
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir ./Wan2.2-S2V-14B
⚠️   Warning: 'huggingface-cli download' is deprecated. Use 'hf download' instead.
```

## snapshots
```python
get_local_snapshots(
    model_id:               # list snapshots with id, checks if a newer one exists in huggingface
    cache_home: Union[str, list, tuple, None] = None, # custom cache homes:  
    download: bool = False, # if True and tag not found, downlaods it
    ) -> list
```

```python
list_local_snapshots(cache_home: Union[str, list, tuple, None] = None, verobse=True) -> dict:
    # list all local snapshots 
```
managed variant of native hf commend 
`$ huggingface-cli scan-cache `

## Pipelines

### diffusers
```python
list_pipelines(key, i=False, w=False) # list available pipelines with grep like switches -i -w
# e.g.
>>> list_pipelines("stableaudio", i=True) # -> ['StableAudioPipeline', 'StableAudioProjectionModel']
```
```python
load_pipeline(pipeline: str, model_id: str, download=False, **kwargs)  # download=False : only load pipe if locally found
# e.g.
>>> pipe = load_pipeline(pipeline='StableDiffusionImg2ImgPipeline', model_id='stable-diffusion-v1-5/stable-diffusion-v1-5', torch_dtype=torch.float16)
>>> pipe = load_pipeline(pipeline='StableDiffusionInpaintPipeline', model_id='stable-diffusion-v1-5/stable-diffusion-inpainting', torch_dtype=torch.float16)

```

### get pipeline for a downloaded snapshot
``` python
get_snapshot_pipeline(snapshot)
# pulls info from model if model_index.json found under snapshot
# Q: does every snapshot have a pipeline ? naa
```
### get snapshots for a pipeline
```python
get_pipeline_snapshots()
```

# Conda / Mamba / Pip helper
In `./conda`, move to a location within `$PATH`,  maybe `~/.local/bin`

* `$ pipinstall <args> <pkg>` # Replaces  `pip install -c $CONDA_PREFIX/constraints.txt <args> <pkg> `
* `$ pin <pkg1> <pkg2> ...` # finds versions of pkgs and pins them both to conda (`$CONDA_PREFIX/conda-meta/pinned`) and pip (`CONDA_PREFIX/constraints.txt`)
* `python envlist.py [rebuild]` # stores `<CONDA ROOT>/conda_envs.csv` with a list of packages of concern to list, pops it in browser. Requires streamlit fpr viewing.
* `streamlit viewenvs.py` # pops  `<CONDA ROOT>/conda_envs.csv` in browser, called by `envlist.py`


# docjumble / Attention
* [Attention](docjumble/Attention.md)
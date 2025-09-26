# hug
basic hugging face handling scripts. Hugging space api is not clear to me, as like tha rest of the AI dev space has lots of code rot.


HF will require tokens and login to download  `huggingface-cli login`

## Hugging face cache
* Local machine default `~/.cache/huggingface`, or `$HUGGINGFACE_HOME` 
* subfolders: models now got to the `<cachehome>/hub`, before to `<>/diffusers` and `<>/transformers`
## Traversing huggingface with cli
* `models = list_models( filter='', search='', sort='', limit='', ... ) -> iterator` [HF Notes](HFNOTES.md) # 2025.09 -> 2M+ models
* `mod = next(models)` -> `.id` (name)==`.modelId` (name), `._id` (snapshot hash), `.pipeline_tag` (task), `.private` (bool), `.tags` (metadata list, papers, etc)

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

## diffusers
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

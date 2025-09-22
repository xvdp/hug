# hug
basic hugging face handling scripts

As with the rest of the ai space, updates mean that code rots quite quickly. 

HF is great in that it is has lots of stuff, but its indirection is a bit annoying.

By default on local machines code gets downloaded to `~/.cache/huggingface`. Fortunately HF provides an env `$HUGGINGFACE_HOME`, which if set will serve as main store. Unfortunately substores keep changing... erst `$HUGGINGFACE_HOME/diffusers` and `$HUGGINGFACE_HOME/transformers` now go to `$HUGGINGFACE_HOME/hub` so you may end up downloading the same projects twice if you arent careful.


## some cmds
Operate on `$HUGGINGFACE_HOME` or `~/.cache/huggingface` or custom cache_home/s, under subfolders /hub /transformers /diffusers
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

```python
list_pipelines(key, i=False, w=False) # list available pipelines with grep like switches -i -w
load_pipeline(pipeline: str, model_id: str, dowmload=False, **kwargs) 
# generic pipeline load, prefer local
```

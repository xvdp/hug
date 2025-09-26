# huggingface hub commands

* `list_models( kwargs ) -> iterator`
    *  `model -> dict` : 

```python
>>> from huggingface_hub import list_models
>>> help(list_models)

    filter: 'Union[str, Iterable[str], None]' = None,
    author: 'Optional[str]' = None,
    apps: 'Optional[Union[str, List[str]]]' = None,
    gated: 'Optional[bool]' = None,
    inference: "Optional[Literal['warm']]" = None,
    inference_provider: "Optional[Union[Literal['all'], 'PROVIDER_T', List['PROVIDER_T']]]" = None,
    model_name: 'Optional[str]' = None,
    trained_dataset: 'Optional[Union[str, List[str]]]' = None,
    search: 'Optional[str]' = None,
    pipeline_tag: 'Optional[str]' = None,
    emissions_thresholds: 'Optional[Tuple[float, float]]' = None,
    sort: "Union[Literal['last_modified'], str, None]" = None,
    direction: 'Optional[Literal[-1]]' = None,
    limit: 'Optional[int]' = None,
    expand: 'Optional[List[ExpandModelProperty_T]]' = None,
    full: 'Optional[bool]' = None,
    cardData: 'bool' = False,
    fetch_config: 'bool' = False,
    token: 'Union[bool, str, None]' = None,
    language: 'Optional[Union[str, List[str]]]' = None,
    library: 'Optional[Union[str, List[str]]]' = None,
    tags: 'Optional[Union[str, List[str]]]' = None,
    # task: 'Optional[Union[str, List[str]]]' = None # will be deprecated use filter instead

# models returns an iterator
>>> models = list_models(filter="diffusers",
                         search="stabilityai/stable-diffusion",
                         sort="downloads",
                         direction=-1,
                         limit=40)
m = next(models, None)
if m:
    m.library_name  # 'diffusers'
    m.modelId       # 'stabilityai/stable-diffusion-2-1'

m.__dict__.keys()
dict_keys(['id', 'author', 'sha', 'last_modified', 'created_at', 'private', 'gated', 'disabled', 'downloads', 'downloads_all_time', 'likes', 'library_name', 'gguf', 'inference', 'inference_provider_mapping', 'tags', 'pipeline_tag', 'mask_token', 'trending_score', 'card_data', 'widget_data', 'model_index', 'config', 'transformers_info', 'siblings', 'spaces', 'safetensors', 'security_repo_status', 'xet_enabled', 'lastModified', 'cardData', 'transformersInfo', '_id', 'modelId'])

# not all keys need to be set, for examplem 
for k, v in m.__dict__.items():
    if v is not None:
        print(f"{k:<20} {v}")
# []
    id                   stabilityai/stable-diffusion-2-1
    created_at           2022-12-06 17:24:51+00:00
    private              False
    downloads            812977
    likes                4026
    library_name         diffusers
    tags                 ['diffusers', 'safetensors', 'stable-diffusion', 'text-to-image', 'arxiv:2112.10752', 'arxiv:2202.00512', 'arxiv:1910.09700', 'license:openrail++', 'autotrain_compatible', 'endpoints_compatible', 'diffusers:StableDiffusionPipeline', 'region:us']
    pipeline_tag         text-to-image
    _id                  638f7ae36c25af4071044105
    modelId              stabilityai/stable-diffusion-2-1



help(m)


likes (`int`):
 |          Number of likes of the model.
 |      library_name (`str`, *optional*):
 |          Library associated with the model.
 |      tags (`List[str]`):
 |          List of tags of the model. Compared to `card_data.tags`, contains extra tags computed by the Hub
 |          (e.g. supported libraries, model's arXiv).
 |      pipeline_tag (`str`, *optional*):
 |          Pipeline tag associated with the model.
 |      mask_token (`str`, *optional*):
 |          Mask token used by the model.
 |      widget_data (`Any`, *optional*):
 |          Widget data associated with the model.
 |      model_index (`Dict`, *optional*):
 |          Model index for evaluation.
 |      config (`Dict`, *optional*):
 |          Model configuration.
 |      transformers_info (`TransformersInfo`, *optional*):
 |          Transformers-specific info (auto class, processor, etc.) associated with the model.
 |      trending_score (`int`, *optional*):
 |          Trending score of the model.
 |      card_data (`ModelCardData`, *optional*):
 |          Model Card Metadata  as a [`huggingface_hub.repocard_data.ModelCardData`] object.
 |      siblings (`List[RepoSibling]`):
 |          List of [`huggingface_hub.hf_api.RepoSibling`] objects that constitute the model.
 |      spaces (`List[str]`, *optional*):
 |          List of spaces using the model.
 |      safetensors (`SafeTensorsInfo`, *optional*):
 |          Model's safetensors information.
 |      security_repo_status (`Dict`, *optional*):
 |          Model's security scan status.
 |
```
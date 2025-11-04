from transformers.models.auto import tokenization_auto, configuration_auto
from huggingface_hub import HfApi


# def list_tokenizers():
#     """using AutoTokenizer"""
#     mapping = tokenization_auto.TOKENIZER_MAPPING
#     # Extract model types
#     models = sorted(tokenization_auto.TOKENIZER_MAPPING.keys())
#     return models

def get_transfomers():
    return dict(configuration_auto.CONFIG_MAPPING)

def list_transformers():
    """using AutoConfig"""
    return sorted(get_transfomers().keys())

def get_checkpoints(model_type=None, limit=10):
    """         filter (`str` or `Iterable[str]`, *optional*):
            A string or list of string to filter models on the Hub.
            Models can be filtered by library, language, task, tags, and more.
    """
    api = HfApi()
    model_types = sorted(configuration_auto.CONFIG_MAPPING.keys())
    if model_type is not None:
        assert model_type in model_types, f"model type {model_type} not found in {model_types}"
    out = {}
    for t in model_types:
        models = api.list_models(filter={"config.model_type": t}, library="transformers", limit=limit)
        out[t] = [m.modelId for m in models]

def describe_checkpoint(model_id):
    api = HfApi()
    info = api.model_info(model_id)
    return {
        "model_id": model_id,
        "private": info.private,
        "gated": info.gated,
        "license": info.cardData.get("license") if info.cardData else None,
    }
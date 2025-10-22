"""
Multipe ways to inspect a torch module


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda:1"

from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16).to("cuda:1")


pprint(pipe.config)
FrozenDict([('vae', ('diffusers', 'AutoencoderKL')),
            ('text_encoder', ('transformers', 'CLIPTextModel')),
            ('text_encoder_2', ('transformers', 'CLIPTextModelWithProjection')),
            ('tokenizer', ('transformers', 'CLIPTokenizer')),
            ('tokenizer_2', ('transformers', 'CLIPTokenizer')),
            ('unet', ('diffusers', 'UNet2DConditionModel')),
            ('scheduler', ('diffusers', 'EulerDiscreteScheduler')),
            ('image_encoder', (None, None)),
            ('feature_extractor', (None, None)),
            ('force_zeros_for_empty_prompt', True),
            ('_name_or_path', 'stabilityai/stable-diffusion-xl-base-1.0'),
            ('_class_name', 'StableDiffusionXLPipeline'),
            ('_diffusers_version', '0.35.1')])

pprint(dict(pipe.vae.config))
    '_name_or_path': '/home/weights/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/vae',
    in_channels: 3,
    out_channels: 3,
    sample_size: 1024,

# Forward
>>> has_callable(pipe.vae.encoder, "forward")


# Get required arguments
>>> help(pipe.vae.encoder.forward)
    forward(sample: torch.Tensor) -> torch.Tensor method of diffusers.models.autoencoders.vae.Encoder instance
>>> get_args_req(enc.forward)
{'sample': <class 'torch.Tensor'>}


 
pprint(dict(pipe.unet.config))
    ('sample_size', 128),
    ('in_channels', 4),
    ('out_channels', 4),



"""




import inspect
from typing import Union, Callable, Optional, Any


import os
import os.path as osp
import pydoc
import torch
import torchinfo
from torch.fx import symbolic_trace
from pprint import pprint



def record_help(obj: Any, filename: str) -> str:
    """ help(obj) > filename
    """
    basedir = osp.abspath(osp.expanduser(osp.dirname(filename)))
    os.makedirs(basedir, exist_ok=True)
    help_text = pydoc.render_doc(obj, renderer=pydoc.plaintext)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("```\n")
        f.write(help_text)
        f.write("\n```\n")
    return help_text


### 
# Callables of a class
def get_callables(cls, inherit=True):
    """Yield (name, callable) for every callable attribute of *cls*."""
    # inspect.getmembers already filters by cls.__dict__ if inherit=False
    for name, obj in inspect.getmembers(cls, predicate=inspect.isroutine):
        if inherit or name in cls.__dict__:      # own vs inherited
            yield name, obj

def print_callables(cls, inherit=True) -> None:
    """
    Example 
    >>> print_callables(pipe.unet)
    """
    pprint(list(dict(get_callables(cls, inherit=inherit)).keys()))

def has_callable(cls, name) -> bool:
    """ 
    Example
    >>> has_callable(pipe.unet, "forward") # True
    """
    for _name, obj in inspect.getmembers(cls, predicate=inspect.isroutine):
        if _name == name:
            return True
    return False


### Arguments of a callable
def get_args_req(func: Callable) -> dict: # {name: type}
    """ ONLY REquired Arguments
    Example:
    >>> get_args_req(pipe.unet.forward)
        {'sample': <class 'torch.Tensor'>, 'timestep': typing.Union[torch.Tensor, float, int], 'encoder_hidden_states': <class 'torch.Tensor'>
    """
    assert isinstance(func, Callable), f"func must be callable, got {type(func)}"
    out = {}
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        if param.default == inspect.Parameter.empty:
            out[name] = param.annotation if param.annotation != inspect.Parameter.empty else "Any"
    return out

def get_args_optional(func: Callable) -> dict: # {name: (type, default)}
    """ ONLY optional Arguments
    """
    assert isinstance(func, Callable), f"func must be callable, got {type(func)}"
    out = {}
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        if param.default != inspect.Parameter.empty:
            out[name] = (param.annotation if param.annotation != inspect.Parameter.empty else "Any", param.default)
    return out

def get_args(func: Callable, arg_type: str = "required") -> dict:
    """   
    class Parameter(builtins.object)
    .name:       str
    .default:    object | .empty
    .annotation: str | .empty`.
    .kind:       .POSITIONAL_ONLY | .POSITIONAL_OR_KEYWORD | .VAR_POSITIONAL | .KEYWORD_ONLY | Parameter.VAR_KEYWORD
    """
    if arg_type == "required":
        return get_args_req(func)
    elif arg_type == "optional":
        return get_args_optional(func)

    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        annot = param.annotation if param.annotation != inspect.Parameter.empty else "Any",
        val = param.default if param.default != inspect.Parameter.empty else "<REQUIRED>"
        print(f"{name:<40}{str(val):<40}{annot}")


def print_config(mod):
    if hasattr(mod, "config"):
        for k,v in mod.config.items():
            print(k, v)

def _dummy_input(mod, input_shape: Union[int, list, tuple], name="module", dtype=None, device=None, **kwargs)-> Union[list, torch.Tensor]:
    """ builds a randn tensor
    """
    def _get_dtype(mod, name="module", dtype=None):
        if dtype is not None:
            assert isinstance(dtype, torch.dtype), f"pass torch dtype, got {type(dtype)}"
        elif hasattr(mod, "dtype"):
            dtype = mod.dtype
        else:
            assert hasattr(mod, "_parameters"), f"module {name} has no parameters"
            dtype = next(iter(mod.parameters())).dtype
        return dtype
    def _get_device(mod, name="module", device=None):
        if device is not None:
            device = torch.device(device)
        elif hasattr(mod, "device"):
            device = mod.device
        else:
            assert hasattr(mod, "_parameters"), f"module {name} has no parameters"
            device = next(iter(mod.parameters())).device
        return device
    def _get_rand_tensor(input_shape, dtype=dtype, device=device, **kwargs):
        if dtype.is_floating_point or dtype.is_complex:
            return torch.randn(input_shape, dtype=dtype, device=device)
        low = kwargs.pop("low", 0)
        high = kwargs.pop("high", 255)
        return torch.randint(low, high, input_shape, dtype=dtype, device=device)
    assert isinstance(input_shape, (int, list, tuple)), f"expected, int, list, tuple) got {type(input_shape)}"
    dtype = _get_dtype(mod, name, dtype)
    device = _get_device(mod, name, device)
    if isinstance(input_shape, int):
        input_shape = [input_shape]
    if isinstance(input_shape[0], (list, tuple)):
        out = []
        for shape in input_shape:
            out.append(_dummy_input(mod, shape, name=name, dtype=dtype, device=device, **kwargs))
        return out
    assert all([isinstance(i, int) for i in input_shape]), f"expexts a list or tuple of ints, got {input_shape}"
    return _get_rand_tensor(input_shape, dtype, device, **kwargs)

def summarize_module(mod: Callable,
                     input_shape: Union[list, tuple, int],
                     name: str = "module",
                     out_file: Optional[str] = None) -> str:
    """Returns a markdown table with in/out shapes for every child.

    summarize_module(pipe.vae.encoder, (1,3,1024,1024), "vae.encoder")
    """
    x = _dummy_input(mod, input_shape, name)
    col = torchinfo.summary(
        mod,
        input_data=x,
        verbose=0,          # silent
        col_names=("input_size", "output_size", "num_params"),
        row_settings=("depth",),
    )
    # torchinfo gives a pretty string – wrap it in markdown
    info = f"### {name}\n```\n{col}\n```\n"
    if out_file is not None:
        if not out_file.endswith[".md"]:
            out_file = out_file+".md"
        with open(out_file, "w") as f:
            f.write(info)
            f.write("\n")


def module_io(mod, input_shape, name="module"):
    "this shoud be for shape in  in input shapes"
    x = _dummy_input(mod, input_shape, name)
    t = _dummy_input(mod, [1], name)
    with torch.no_grad():
        y = mod(x, t)   # or add encoder_hidden_states=… for SDXL
    print(f"{name} input :", x.shape, x.dtype)
    print(f"{name} output:", y.sample.shape, y.sample.dtype)


### dont quite work yet
def module_trace(mod, input_shape, name="module"):
    x = _dummy_input(mod, input_shape, name)
    graph = symbolic_trace(mod, concrete_args={"sample": x})
    graph.graph.print_tabular()


def module_export(mod, input_shape, name="module"):
    x = _dummy_input(mod, input_shape, name)
    exported = torch.export.export(mod,(x,))
    print(exported.graph)          # full FX graph, no control-flow issues



def unet_dummies(unet, batch=1, dtype=torch.float16, device="cuda"):
    c = unet.config
    h = w = c.sample_size
    return dict(
        sample               = torch.randn(batch, c.in_channels, h, w, dtype=dtype, device=device),
        timestep             = torch.tensor(500, dtype=dtype, device=device),
        encoder_hidden_states= torch.randn(batch, 77, c.cross_attention_dim, dtype=dtype, device=device),
        added_cond_kwargs    = {          # SDXL only
            "text_embeds":     torch.randn(batch, 1280, dtype=dtype, device=device),
            "time_ids":        torch.randn(batch, 6,   dtype=dtype, device=device),
        }
    )

# dummies = unet_dummies(pipe.unet)
# out = pipe.unet(**dummies)
# print("UNet output shape:", out.sample.shape)


# # Example for VAE encoder (half-precision)
# md_vae = summarize_module(
#     pipe.vae.encoder,
#     input_shape=(1, 3, 1024, 1024),   # (B, C, H, W)
#     name="VAE encoder"
# )

# # Example for UNet (SDXL cross-attn)
# md_unet = summarize_module(
#     pipe.unet,
#     input_shape=(1, 4, 128, 128),     # latent space
#     name="UNet"
# )

# with open("shape_report.md", "w") as f:
#     f.write(md_vae)
#     f.write("\n")
#     f.write(md_unet)
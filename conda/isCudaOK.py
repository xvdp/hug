#!/usr/bin/env python3
"""
check-cuda-alignment.py
prints:
  CUDA_HOME        : 13.0
  nvcc             : 13.0
  torch.version.cuda: 12.8
and a short conclusion.
"""
import json, os, re, subprocess, sys

def ver(s: str) -> str:
    """Keep only major.minor"""
    m = re.search(r'(\d+\.\d+)', s or '')
    return m.group(1) if m else None

def printv(*args, **kwargs):
    if kwargs.pop('verbose', True):
        print(*args, **kwargs)

def get_cuda_ok(verbose=False):
    """
    compares cuda versions in current home
    """
    # 1. CUDA_HOME
    
    cuda_home_ver = None
    cuda_home = os.environ.get("CUDA_HOME", "")
    version_file = os.path.join(cuda_home, "version.json")
    if cuda_home and os.path.isfile(version_file):
        try:
            with open(version_file) as f:
                data = json.load(f)
            # prefer top-level "cuda", fallback "cuda_nvcc"
            cuda_home_ver = ver(data.get("cuda", {}).get("version", ""))
            if not cuda_home_ver:
                cuda_home_ver = ver(data.get("cuda_nvcc", {}).get("version", ""))
        except Exception:
            pass
    _cuda_home = f"CUDA_HOME ({cuda_home or 'Not set'}:)"
    printv(f"{_cuda_home:<45} {cuda_home_ver or 'Not found'}", verbose=verbose)

    # 2. nvcc
    nvcc_ver = None
    try:
        txt = subprocess.check_output(["nvcc", "--version"], text=True)
        nvcc_ver = ver(txt)
    except Exception:
        pass

    printv(f"{'nvcc --version:':<45} {nvcc_ver or 'Not found'}", verbose=verbose)

    
    # 3. PyTorch
    torch_ver = None
    try:
        import torch
        torch_ver = ver(torch.version.cuda)
    except Exception:
        pass

    printv(f"{'torch.version.cuda:':<45} {torch_ver or 'Not found'}", verbose=verbose)

    # 4. conclusion
    isOK = False
    if torch_ver and nvcc_ver and cuda_home_ver:
        if torch_ver == nvcc_ver:
            printv(f" =>  nvcc --version === torch.cuda.version {torch_ver}", verbose=verbose)
            if float(cuda_home_ver) >= float(torch_ver):
                printv(" OK: $CUDA_HOME version ≥ runtime version", verbose=verbose)
                isOK = True
            else:
                printv(" Inconsistent versions: $CUDA_HOME version < runtime/compiler version  may fail", verbose=verbose)
        else:
            printv(f" Inconsistent versions: nvcc ({nvcc_ver}) != torch.cuda.version ({torch_ver})", verbose=verbose)
    else:
        printv(" Missing components cannot verify alignment", verbose=verbose)
    return(isOK, {"CUDA_HOME" : cuda_home_ver, "nvcc": nvcc_ver, "torch.cuda" : torch_ver})

if __name__ == "__main__":
    get_cuda_ok(verbose=True)
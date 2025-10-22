

from setuptools import setup
import os
import sys
from importlib.metadata import version, PackageNotFoundError
from packaging.requirements import Requirement

def _set_version(version):
    with open('hug/version.py', 'w', encoding='utf8') as _fi:
        _fi.write("version='"+version+"'")
        return version

def is_conda_environment():
    return os.getenv("CONDA_PREFIX") is not None

def get_installed_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None

def check_for_conflicts(requirements):
    conflicts = []
    for req_str in requirements:
        req = Requirement(req_str)
        installed_version = get_installed_version(req.name)
        if installed_version and not req.specifier.contains(installed_version):
            conflicts.append((req.name, installed_version, str(req.specifier)))
    return conflicts

setup(
    name="hug",
    install_requires=['huggingface_hub', "diffusers"],
    packages=["hug"],
    url='http://github.com/xvdp/hug',
    version=_set_version(version='0.0.3')
)
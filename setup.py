from pathlib import Path
import os
import shutil
import subprocess

from setuptools import setup, find_packages
from torch.utils import cpp_extension

from typing import List

ROOT_DIR = Path(__file__).parent
long_description = (ROOT_DIR/ "README.md").read_text()


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def move_gurobi_license():

    gurobi_license_path = '/opt/gurobi/'
    if not os.path.exists(gurobi_license_path):
        os.mkdir(gurobi_license_path)
    # check if the this_directory/configs/gurobi.lic exists
    gurobi_license_file = ROOT_DIR / 'configs/gurobi.lic'
    if gurobi_license_file.exists():
        shutil.copy(gurobi_license_file, gurobi_license_path)

# requirements
def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements

# install QLLM and LPtorch
def install_customized_packges(requirements):
    # first try install within the current folder
    # check QLLM and LPTorch exists
    if os.path.exists(get_path('3rd_party/QLLM')) and os.path.exists(get_path('3rd_party/QLLM/3rd_party/LPTorch')):
        install_commands = [
            "cd 3rd_party/QLLM/3rd_party/LPTorch && python3 -m pip install -e .",
            "cd 3rd_party/QLLM/3rd_party/transformers && python3 -m pip install -e .",
            "cd 3rd_party/QLLM && python3 -m pip install -e .",
        ]
        for command in install_commands:
            subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    else:
        # if the current folder didn't holds QLLM & LLM-PQ, then install from github
        requirements.append(
            "git+https://github.com/tonyzhao-jt/LPTorch.git",
        )
        requirements.append(
            "git+https://github.com/tonyzhao-jt/QLLM.git"
        )

# update the GPTQ repo for the accuracy
def update_gptq():
    assert os.path.exists(get_path('3rd_party/gptq'))
    install_commands = [
        "cd configs/gptq && bash update.sh",
    ]
    for command in install_commands:
        subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


requirements = get_requirements()
install_customized_packges(requirements)
move_gurobi_license()
update_gptq()

package_name = 'shaq'
setup(
    name=package_name,
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)
    },
    packages=find_packages(
        exclude=['notebook', 'scripts', 'tests', 'bench', 'result']),
    entry_points={
        'console_scripts': [
            f'{package_name}-algo = {package_name}.algo.entry:algo_main',
            f'{package_name}-algo-check = {package_name}.algo.checker:check_strat_file',
            f'{package_name}-algo-cmp = {package_name}.algo.checker:compare_bitwidth_of_two_strat',
            # f'{package_name}-algo-shaqef = {package_name}.algo.shaq_heuristic:shaq_h_main',
            f'{package_name}-algo-shaqef = {package_name}.algo.shaq_efficient:shaq_ef_main',
            f'{package_name}-dist = {package_name}.entrypoints:run_dist',
            f'{package_name}-sole = {package_name}.entrypoints:run_sole'
        ]
    },
    install_requires=requirements,
    # temporaryly write like that, later reconstruct the package to make it as a simple entry points
    package_data={
        f'{package_name}': ['dist_runtime/entry.py', 'dist_runtime/entry_sole.py', 'dist_runtime/utils.py'],
    },
)


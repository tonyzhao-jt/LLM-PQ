from setuptools import setup, find_packages
from torch.utils import cpp_extension

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
            # f'{package_name}-algo-shaqef = {package_name}.algo.shaq_heuristic:shaq_h_main',
            f'{package_name}-algo-shaqef = {package_name}.algo.shaq_efficient:shaq_ef_main',
            f'{package_name}-dist = {package_name}.entrypoints:run_dist'
        ]
    },
    install_requires=[
        'torch'
    ],
    # temporaryly write like that, later reconstruct the package to make it as a simple entry points
    package_data={
        f'{package_name}': ['dist_runtime/entry.py', 'dist_runtime/utils.py'],
    },
)


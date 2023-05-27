from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(
    name='qpipe',
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)
    },
    packages=find_packages(
        exclude=['notebook', 'scripts', 'tests', 'bench', 'result']),
)
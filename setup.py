from setuptools import setup, find_packages
from torch.utils import cpp_extension

package_name = 'qpipe'
setup(
    name=package_name,
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)
    },
    packages=find_packages(
        exclude=['notebook', 'scripts', 'tests', 'bench', 'result']),
    entry_points={
        'console_scripts': [
            f'{package_name} = {package_name}.algo_entry:main'
        ]
    }
)
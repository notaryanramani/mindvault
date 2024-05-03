from setuptools import setup, find_packages
import sys

def get_requirements():
    with open('requirements.txt', 'r') as f:
        req = f.read().splitlines()
    python_version = f'cp{sys.version_info.major}{sys.version_info.minor}'
    if sys.platform == 'win32':
        req.append(f'torch @https://download.pytorch.org/whl/cu121/torch-2.3.0%2Bcu121-{python_version}-{python_version}-win_amd64.whl')
    else:
        req.append('torch')
    return req

setup(
    name='mindvault',
    version='0.1.0',
    author='Aryan Ramani',
    author_email='aryanramani67@gmail.com',
    description='A mini transformer with a memory bank',
    packages=find_packages(),
    install_requires=get_requirements(),
    classifiers=[
        'Programming Language :: Python :: 3.11',
    ],
)
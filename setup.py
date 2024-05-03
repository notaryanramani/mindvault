from setuptools import setup, find_packages

def get_requirements():
    with open('requirements.txt', 'r') as f:
        return f.read().splitlines()

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
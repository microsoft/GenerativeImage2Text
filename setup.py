from setuptools import setup, find_packages

# python setup.py build develop --user

setup(name='generativeimage2text',
      version='0.1',
      description='Reference implementation of GIT model',
      install_requires=open("requirements.txt").read().splitlines(),
      packages=find_packages(),
 )



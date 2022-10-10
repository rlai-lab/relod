from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3 or sys.version_info.minor != 6:
    print(f'This package is only compatible with Python 3.6, but you are running '
          f'Python {sys.version_info.major}.{sys.version_info.minor}.\nExiting installation...')
    exit()

setup(name='relod',
      packages=[package for package in find_packages()
                if package.startswith('relod')],
      description='Real-time Reinforcement Learning for Vision-Based Robotics Utilizing Local and Remote Computers',
      author='RLAI-Lab ReLoD Team',
      url='https://github.com/rlai-lab/relod',
      author_email='yan28@ualberta.ca',
      version='1.0.0')

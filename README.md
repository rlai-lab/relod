# ReLoD: The Remote-Local Distributed System for Real-time Reinforcement Learning on Vision-Based Robotics Tasks

Real-time Reinforcement Learning for Vision-Based Robotics Utilizing Local and Remote Computers

## Installation instructions
1. Download Mujoco150, Mujoco200 and license files to ~/.mujoco
2. Install miniconda or anaconda
3. Create a virtual environment:
```bash
conda create --name myenv python=3.6
conda activate myenv
```
3. Add the following to ~/.bashrc:
```bash
conda activate myenv
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/iqapple/.mujoco/mjpro150/bin
export MUJOCO_GL="egl"
```
and run:
```bash
source ~/.bashrc
```
4. Install packages with:
```bash
pip install -r requirements.txt
pip install .
```

## Cite
```bash

```

# robotis_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

![ISAAC_FFW_RL_reach_train (1)](https://github.com/user-attachments/assets/5ca984e9-acbb-4505-95d8-a2b04d3b3980)

## Overview

**robotis_lab** is a research-oriented repository based on [Isaac Lab](https://isaac-sim.github.io/IsaacLab), designed to enable reinforcement learning (RL) and imitation learning (IL) experiments using Robotis robots in simulation.
This project provides simulation environments, configuration tools, and task definitions tailored for Robotis hardware, leveraging NVIDIA Isaac Sim’s powerful GPU-accelerated physics engine and Isaac Lab’s modular RL pipeline.

> [!IMPORTANT]
> This repository currently depends on **IsaacLab v2.0.0** or higher.
>

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone the robotis_lab Repository (i.e. outside the `IsaacLab` directory):

  ```bash
  git clone https://github.com/ROBOTIS-GIT/robotis_lab.git
  ```

- Install the robotis_lab Package

  ```bash
  cd robotis_lab && python -m pip install -e source/robotis_lab

  ```

- Verify that the extension is correctly installed by running the following command to print all the available environments in the extension:

  ```bash
  python scripts/tools/list_envs.py
  ```

## Try examples

### Sim2Sim
<details>
<summary>Reinforcement learning</summary>

OMY Reach task

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotisLab-Reach-OMY-v0 --num_envs=512 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task RobotisLab-Reach-OMY-v0 --num_envs=16
```

OMY Lift task

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotisLab-Lift-Cube-OMY-v0 --num_envs=512 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task RobotisLab-Lift-Cube-OMY-v0 --num_envs=16
```

OMY Open drawer task

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotisLab-Open-Drawer-OMY-v0 --num_envs=512 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task RobotisLab-Open-Drawer-OMY-v0 --num_envs=16
```

FFW-BG2 reach task

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotisLab-Reach-FFW-BG2-v0 --num_envs=512 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task RobotisLab-Reach-FFW-BG2-v0 --num_envs=16
```

</details>

<details>
<summary>Imitation learning</summary>

> [!NOTE]
> If you want to control a **SINGLE ROBOT** with the keyboard during playback, add `--keyboard` at the end of the play script.
>
> ```
> Key bindings:
> =========================== =========================
> Command                     Key
> =========================== =========================
> Toggle gripper (open/close) K      
> Move arm along x-axis       W / S   
> Move arm along y-axis       A / D
> Move arm along z-axis       Q / E
> Rotate arm along x-axis     Z / X
> Rotate arm along y-axis     T / G
> Rotate arm along z-axis     C / V
> =========================== =========================
> ```

OMY Stack task (Stack the blocks in the following order: blue → red → green.)

```bash
# Teleop and record
python scripts/imitation_learning/isaaclab_recorder/record_demos.py --task RobotisLab-Stack-Cube-OMY-IK-Rel-v0 --teleop_device keyboard --dataset_file ./datasets/dataset.hdf5 --num_demos 10

# Annotate
python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --device cuda --task RobotisLab-Stack-Cube-OMY-IK-Rel-Mimic-v0 --auto --input_file ./datasets/dataset.hdf5 --output_file ./datasets/annotated_dataset.hdf5 --headless

# Mimic data
python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
--device cuda --num_envs 100 --generation_num_trials 1000 \
--input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset.hdf5 --headless

# Train
python scripts/imitation_learning/robomimic/train.py \
--task RobotisLab-Stack-Cube-OMY-IK-Rel-v0 --algo bc \
--dataset ./datasets/generated_dataset.hdf5

# Play
python scripts/imitation_learning/robomimic/play.py \
--device cuda --task RobotisLab-Stack-Cube-OMY-IK-Rel-v0 --num_rollouts 50 \
--checkpoint /PATH/TO/desired_model_checkpoint.pth
```

FFW-BG2 Pick and Place Task (Move the red stick into the basket.)

```bash
# Teleop and record
python scripts/tools/record_demos.py --task RobotisLab-PickPlace-FFW-BG2-IK-Rel-v0 --teleop_device keyboard --dataset_file ./datasets/dataset.hdf5 --num_demos 10 --enable_cameras

# Annotate
python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --device cuda --task RobotisLab-PickPlace-FFW-BG2-Mimic-v0 --input_file ./datasets/dataset.hdf5 --output_file ./datasets/annotated_dataset.hdf5 --enable_cameras

# Mimic data
python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
--device cuda --num_envs 20 --generation_num_trials 300 \
--input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset.hdf5 --enable_cameras --headless

# Train
python scripts/imitation_learning/robomimic/train.py \
--task RobotisLab-PickPlace-FFW-BG2-IK-Rel-v0 --algo bc \
--dataset ./datasets/generated_dataset.hdf5

# Play
python scripts/imitation_learning/robomimic/play.py \
--device cuda --task RobotisLab-PickPlace-FFW-BG2-IK-Rel-v0  --num_rollouts 50 \
--checkpoint /PATH/TO/desired_model_checkpoint.pth --enable_cameras
```
</details>

### Sim2Real

<details>
<summary>Reinforcement learning</summary>

https://github.com/user-attachments/assets/6c27bdb1-3a6b-4686-a546-8f14f01e4abe


> [!IMPORTANT]
> More on OMY Hardware Setup:
> For details on how to set up and operate the OMY robot, please refer to the [open_manipulator repo](https://github.com/ROBOTIS-GIT/open_manipulator.git)

In this pipeline:
- The trained policy (exported as a TorchScript .pt file) is executed on the real robot using ROS 2.
- The robot receives joint state feedback and sends joint trajectory commands via a ROS 2 control interface.
- A TF frame for the sampled target pose is broadcast for visualization and debugging.

Prerequisites
- A trained policy (under logs/rsl_rl/reach_omy/<TIMESTAMP>).
- ROS 2 Jazzy installed and sourced.
- Robot hardware must be ready and controllable via the joint trajectory interface.


Run Sim2Real Reach Policy on OMY

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotisLab-Reach-OMY-v0 --num_envs=512 --headless

# Sim2Real
python scripts/sim2real/reinforcement_learning/inference/OMY/reach/run_omy_reach.py --model_dir=<2025-07-10_08-47-09>
```

Replace <2025-07-10_08-47-09> with the actual timestamp folder name under:
```bash
logs/rsl_rl/reach_omy/
```
</details>

<details>
<summary>Imitation learning</summary>

```bash

# install
pip install lerobot

# Teleop and record
python scripts/sim2real/imitation_learning/recorder/record_demos.py --task=RobotisLab-Real-Pickup-Bottle-OMY-v0 --robot_type OMY --dataset_file ./datasets/omy_pickup_task.hdf5 --num_demos 30 --enable_cameras

```

<details>
<summary>[Option] Mimic generate dataset</summary>

```bash

# Data convert ee_pose action from joint action
python scripts/sim2real/imitation_learning/mimic/action_data_converter.py --input_file ./datasets/omy_pickup_task.hdf5 --output_file ./datasets/processed_omy_pickup_task.hdf5 --action_type ik

# Annotate dataset
python scripts/sim2real/imitation_learning/mimic/annotate_demos.py --task RobotisLab-Real-Mimic-Pickup-Bottle-OMY-v0 --auto --input_file ./datasets/processed_omy_pickup_task.hdf5 --output_file ./datasets/annotated_dataset.hdf5 --enable_cameras --headless

# Generate dataset
python scripts/sim2real/imitation_learning/mimic/generate_dataset.py --device cuda --num_envs 30 --task RobotisLab-Real-Mimic-Pickup-Bottle-OMY-v0 --generation_num_trials 1000 --input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset.hdf5 --enable_cameras --headless

# Data convert joint action from ee_pose action
python scripts/sim2real/imitation_learning/mimic/action_data_converter.py --input_file ./datasets/generated_dataset.hdf5 --output_file ./datasets/processed_generated_dataset.hdf5 --action_type joint

```

</details>

```bash

# Data convert lerobot dataset from IsaacLab hdf dataset
python scripts/sim2real/imitation_learning/data_converter/OMY/isaaclab2lerobot.py --task=RobotisLab-Real-Pickup-Bottle-OMY-v0 --robot_type OMY --dataset_file ./datasets/<processed_omy_pickup_task.hdf5> or <processed_generated_dataset.hdf5>

```

</details>

## License

This repository is licensed under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.

### Third-party components

- **Isaac Lab**: BSD-3-Clause License, see [LICENSE-IsaacLab](LICENSE-IsaacLab)

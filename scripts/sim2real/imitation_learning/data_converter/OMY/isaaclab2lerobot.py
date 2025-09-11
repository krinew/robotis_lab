# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import h5py
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime

from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAMERA_CONFIG = {
    "cam_wrist": {"height": 424, "width": 240},
    "cam_top": {"height": 424, "width": 240},
}

def get_env_features(fps: int, camera_config=CAMERA_CONFIG):
    features = {
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "joint1.pos", "joint2.pos", "joint3.pos", "joint4.pos",
                "joint5.pos", "joint6.pos", "rh_r1_joint.pos",
            ]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "joint1.pos", "joint2.pos", "joint3.pos", "joint4.pos",
                "joint5.pos", "joint6.pos", "rh_r1_joint.pos",
            ]
        }
    }

    for cam_name, cfg in camera_config.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [cfg["height"], cfg["width"], 3],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.height": cfg["height"],
                "video.width": cfg["width"],
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": float(fps),
                "video.channels": 3,
                "has_audio": False,
            },
        }

    return features

def process_data(dataset: LeRobotDataset, task: str, demo_group: h5py.Group, demo_name: str) -> bool:
    """
    Process a single demonstration group from the HDF5 dataset
    and add it into the LeRobot dataset.
    """
    try:
        # Load entire data arrays from HDF5
        actions = np.array(demo_group['obs/actions'], dtype=np.float32)
        joint_pos = np.array(demo_group['obs/joint_pos'], dtype=np.float32)
        cam_wrist_images = np.array(demo_group['obs/cam_wrist'], dtype=np.uint8)
        cam_top_images = np.array(demo_group['obs/cam_top'], dtype=np.uint8)
    except KeyError:
        print(f"Demo {demo_name} is not valid, skipping...")
        return False

    # Ensure actions and joint positions are 2D arrays
    if actions.ndim == 1:
        actions = actions.reshape(-1, 7)
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.reshape(-1, 7)
    
    total_state_frames = actions.shape[0]

    # Process each frame
    for frame_index in tqdm(range(total_state_frames), desc=f"Processing demo {demo_name}"):
        frame = {
            "action": actions[frame_index],
            "observation.state": joint_pos[frame_index],
            "observation.images.cam_wrist": cam_wrist_images[frame_index],
            "observation.images.cam_top": cam_top_images[frame_index],
        }
        dataset.add_frame(frame=frame, task=task)

    return True


def convert_isaaclab_to_lerobot(
    task: str, repo_id: str, robot_type: str, dataset_file: str,
    fps: int, push_to_hub: bool = False
):
    """
    Convert an IsaacLab HDF5 dataset into LeRobot dataset format.
    """
    hdf5_files = [dataset_file]
    now_episode_index = 0

    # Create a new LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=get_env_features(fps),
    )

    # Process each HDF5 dataset file
    for hdf5_id, hdf5_file in enumerate(hdf5_files):
        print(f"[{hdf5_id+1}/{len(hdf5_files)}] Processing HDF5 file: {hdf5_file}")
        with h5py.File(hdf5_file, "r") as f:
            demo_names = list(f["data"].keys())
            print(f"Found {len(demo_names)} demos: {demo_names}")

            for demo_name in tqdm(demo_names, desc="Processing each demo"):
                demo_group = f["data"][demo_name]

                # Skip unsuccessful demonstrations
                if "success" in demo_group.attrs and not demo_group.attrs["success"]:
                    print(f"Demo {demo_name} not successful, skipping...")
                    continue

                valid = process_data(dataset, task, demo_group, demo_name)

                if valid:
                    now_episode_index += 1
                    dataset.save_episode()
                    print(f"Saved episode {now_episode_index} successfully")

    # Optionally push to HuggingFace Hub
    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IsaacLab dataset to LeRobot format")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., OMY_Pickup)")
    parser.add_argument("--robot_type", type=str, default="OMY", help="Robot type (default: OMY)")
    parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="Path to dataset HDF5 file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for dataset (default: 30)")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push dataset to HuggingFace Hub")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    default_repo_id = f"data/{timestamp}"
    parser.add_argument("--repo_id", type=str, default=default_repo_id, help=f"Repo ID (default: {default_repo_id})")

    args = parser.parse_args()

    convert_isaaclab_to_lerobot(
        task=args.task,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        dataset_file=args.dataset_file,
        fps=args.fps,
        push_to_hub=args.push_to_hub
    )

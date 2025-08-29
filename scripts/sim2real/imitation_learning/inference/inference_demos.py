# inference.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run inference with robotis_lab environments using OMYLeader."""

import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Inference script for robotis_lab environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--robot_type", type=str, default="OMY", choices=['OMY'], help="Type of robot to use for teleoperation.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import time
import torch
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import robotis_lab

class RateLimiter:
    """Simple class for enforcing a loop frequency."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration

def main():
    # env config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed

    # create env
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # teleop interface
    if args_cli.robot_type == "OMY":
        from dds_sdk.omy_sdk import OMYSdk
        teleop_interface = OMYSdk(env, mode='inference')
    else:
        raise ValueError(f"Unsupported robot type: {args_cli.robot_type}")

    # reset env
    env.reset()
    teleop_interface.reset()
    rate_limiter = RateLimiter(args_cli.step_hz)

    print("[INFO] Inference loop started. Press 'R' to reset environment.")
    should_reset_task = False
    def reset_task():
        nonlocal should_reset_task
        should_reset_task = True

    teleop_interface.add_callback("R", reset_task)

    while simulation_app.is_running():
        with torch.inference_mode():
            # Always publish observations (images and joint states)
            teleop_interface.publish_observations()
            actions = teleop_interface.get_action()

            if should_reset_task:
                print("[INFO] Reset requested.")
                should_reset_task = False
                env.reset()
                continue

            elif actions is None:
                env.render()
            else:
                env.step(actions)
            if rate_limiter:
                rate_limiter.sleep(env)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

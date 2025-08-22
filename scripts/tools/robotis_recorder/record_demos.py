# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a robotis_lab teleoperation with robotis_lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="robotis_lab teleoperation for robotis_lab environments.")
parser.add_argument("--teleop_device", type=str, default="keyboard", choices=['omy_leader'], help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")

# recorder_parameter
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos.")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
import time
import torch
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import TerminationTermCfg

from dds_sdk.omy_leader import OMYLeader

import robotis_lab
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "recorder_manager"))
from recorder_manager import StreamingRecorderManager

class RateLimiter:
    """Convenience class for enforcing rates in loops."""

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
    """Running robotis_lab teleoperation with robotis_lab manipulation environment."""

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.use_teleop_device(args_cli.teleop_device)
    env_cfg.seed = args_cli.seed
    task_name = args_cli.task

    # modify configuration
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    if not hasattr(env_cfg.terminations, "success"):
        setattr(env_cfg.terminations, "success", None)
    env_cfg.terminations.success = TerminationTermCfg(func=lambda env: torch.zeros(1, dtype=torch.bool, device=env.device))

    # create environment
    env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    del env.recorder_manager
    env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
    env.recorder_manager.flush_steps = 100
    env.recorder_manager.compression = 'lzf'

    # create controller
    if args_cli.teleop_device == "omy_leader":
        teleop_interface = OMYLeader(env)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'omy_leader'."
        )

    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # add teleoperation key for task success
    should_reset_task_success = False

    def reset_task_success():
        nonlocal should_reset_task_success
        should_reset_task_success = True
        reset_recording_instance()

    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.add_callback("N", reset_task_success)
    print(teleop_interface)

    rate_limiter = RateLimiter(args_cli.step_hz)

    # reset environment
    env.reset()
    teleop_interface.reset()

    current_recorded_demo_count = 0
    current_failed_demo_count = 0

    start_record_state = False

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            actions = teleop_interface.advance()
            if should_reset_task_success:
                print("Task Success!!!")
                should_reset_task_success = False
                env.termination_manager.set_term_cfg("success", TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)))
                env.termination_manager.compute()
            if should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False
                if start_record_state:
                    print("Stop Recording!!!")
                start_record_state = False
                env.termination_manager.set_term_cfg("success", TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)))
                # print out the current demo count if it has changed
                print(f"Resetting recording instance. Current recorded demo count: {current_recorded_demo_count}")
                if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                    current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                    print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                    print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                    break

            elif actions is None:
                env.render()
            # apply actions
            else:
                if not start_record_state:
                    print("Start Recording!!!")
                    start_record_state = True
                env.step(actions)
            if rate_limiter:
                rate_limiter.sleep(env)

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()

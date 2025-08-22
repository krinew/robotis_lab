import os
import threading
import torch
import numpy as np
from pynput.keyboard import Listener
from collections.abc import Callable

from robotis_python_sdk.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from robotis_python_sdk.idl.trajectory_msgs.msg.dds_ import JointTrajectory_

class OMYLeader:
    """OMYLeader class for receiving joint trajectory updates via ROS2 DDS."""

    def __init__(self, env):
        self.env = env
        self.latest_positions = None
        self.running = True
        self.domain_id = int(os.getenv("ROS_DOMAIN_ID", 0))

        # Final 10 joints to be passed into IsaacLab action
        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
            'rh_r1_joint', 'rh_l1', 'rh_l2', 'rh_r2'
        ]

        # DDS initialization
        ChannelFactoryInitialize(id=self.domain_id)
        self.sub = ChannelSubscriber("rt/leader/joint_trajectory", JointTrajectory_)
        self.sub.Init()

        # Start subscription thread
        self.thread = threading.Thread(target=self._subscriber_loop, daemon=True)
        self.thread.start()

        # some flags and callbacks
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self._display_controls()

    def _display_controls(self):
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("b", "start control")
        print_command("r", "reset simulation and set task success to False")
        print_command("n", "reset simulation and set task success to True")
        print_command("move leader", "control follower in the simulation")
        print_command("Control+C", "quit")
        print("")

    def on_press(self, key):
        pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """
        try:
            if key.char == 'b':
                self._started = True
                self._reset_state = False
            elif key.char == 'r':
                self._started = False
                self._reset_state = True
                self._additional_callbacks["R"]()
            elif key.char == 'n':
                self._started = False
                self._reset_state = True
                self._additional_callbacks["N"]()
        except AttributeError:
            pass

    def _subscriber_loop(self):
        """Subscriber thread: updates the latest joint positions"""
        try:
            while self.running:
                msg = self.sub.Read()  # blocking
                if msg is not None and msg.points:
                    msg_joint_names = msg.joint_names
                    msg_positions = msg.points[-1].positions

                    joint_dict = dict(zip(msg_joint_names, msg_positions))

                    # Fill mimic joint values based on rh_r1_joint
                    master_val = joint_dict.get("rh_r1_joint", 0.0)
                    joint_dict["rh_l1"] = master_val
                    joint_dict["rh_l2"] = master_val
                    joint_dict["rh_r2"] = master_val

                    # Reorder according to self.joint_names
                    self.latest_positions = [joint_dict.get(name, 0.0) for name in self.joint_names]

        except Exception as e:
            print("Subscriber thread exception:", e)
        finally:
            self.sub.Close()
            print("Subscriber closed")

    def get_device_state(self):
        if self.latest_positions is None:
            return {name: 0.0 for name in self.joint_names}
        return {name: pos for name, pos in zip(self.joint_names, self.latest_positions)}

    def shutdown(self):
        self.running = False
        self.thread.join()

    def reset(self):
        pass

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def input2action(self):
        state = {}
        reset = state["reset"] = self._reset_state
        state['started'] = self._started
        if reset:
            self._reset_state = False
            return state
        state['joint_state'] = self.get_device_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict['started'] = self._started
        ac_dict['omy_leader'] = True
        if reset:
            return ac_dict
        ac_dict['joint_state'] = state['joint_state']
        return ac_dict

    def advance(self):
        """
        Returns:
            Can be:
                - torch.Tensor: The action to be applied to the robot.
                - dict: state of the scene and the task, and the task need to reset.
                - None: the scene is not started
        """
        action = self.input2action()
        if action is None:
            return self.env.action_manager.action
        if not action['started']:
            return None
        if action['reset']:
            return action
        for key, value in action.items():
            if isinstance(value, np.ndarray):
                action[key] = torch.tensor(value, device=self.env.device, dtype=torch.float32)
        return self.preprocess_device_action(action)


    def preprocess_device_action(self, action: dict) -> torch.Tensor:
        if action.get('omy_leader'):
            joint_state = action['joint_state']
            positions = [joint_state.get(name, 0.0) for name in self.joint_names]

            processed_action = torch.tensor(
                positions,
                device=self.env.device,
                dtype=torch.float32
            ).unsqueeze(0)

            return processed_action
        else:
            raise NotImplementedError("Only teleoperation with omy_leader is supported for now.")

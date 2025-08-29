import os
import threading
import torch
import cv2
import numpy as np
from pynput.keyboard import Listener
from collections.abc import Callable
from datetime import datetime
from robotis_python_sdk.core.channel import (
    ChannelSubscriber,
    ChannelPublisher,
    ChannelFactoryInitialize,
)
from robotis_python_sdk.idl.trajectory_msgs.msg.dds_ import JointTrajectory_
from robotis_python_sdk.idl.sensor_msgs.msg.dds_ import JointState_
from robotis_python_sdk.idl.sensor_msgs.msg.dds_ import CompressedImage_
from robotis_python_sdk.idl.std_msgs.msg.dds_ import Header_
from robotis_python_sdk.idl.builtin_interfaces.msg.dds_ import Time_

class OMYSdk:
    """OMYSdk class for DDS teleoperation + publishing state/image."""

    def __init__(self, env, mode):
        self.env = env
        self.joint_trajectory_cmd = None
        self.running = True
        self.domain_id = int(os.getenv("ROS_DOMAIN_ID", 0))
        self.mode = mode  # 'record' or 'inference'

        # Joint names
        self.joint_names = self.env.scene["robot"].data.joint_names
        self.exclude_joints = []

        # Initialize DDS
        ChannelFactoryInitialize(id=self.domain_id)

        # Subscriber (leader input)
        self.sub = ChannelSubscriber("rt/leader/joint_trajectory", JointTrajectory_)
        self.sub.Init()

        # Publishers (state + camera)
        self.pub_joint = ChannelPublisher("rt/joint_states", JointState_)
        self.pub_joint.Init()

        self.pub_top_cam = ChannelPublisher("rt/camera/cam_top/color/image_rect_raw/compressed", CompressedImage_)
        self.pub_top_cam.Init()

        self.pub_wrist_cam = ChannelPublisher("rt/camera/cam_wrist/color/image_rect_raw/compressed", CompressedImage_)
        self.pub_wrist_cam.Init()

        # Subscriber thread
        self.thread = threading.Thread(target=self._subscriber_loop, daemon=True)
        self.thread.start()

        # Flags / callbacks
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}

        # Keyboard listener
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self._display_controls()

    def _display_controls(self):
        print("\n[R] Reset simulation (observations continue)")
        print("[B] Start/Resume robot control")
        print("Note: Images and joint states are always published\n")

    def on_press(self, key):  # not used
        pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """
        try:
            if self.mode =='record':
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
            elif self.mode == 'inference':
                if key.char == 'b':
                    self._started = True
                    self._reset_state = False
                elif key.char == 'r':
                    self._started = False
                    self._reset_state = True
                    self._additional_callbacks["R"]()
        except AttributeError:
            pass

    def _subscriber_loop(self):
        try:
            while self.running:
                msg = self.sub.Read()  # blocking read
                if msg is not None and msg.points:
                    msg_joint_names = msg.joint_names
                    msg_positions = msg.points[-1].positions

                    joint_dict = dict(zip(msg_joint_names, msg_positions))
                    master_val = joint_dict.get("rh_r1_joint", 0.0)
                    joint_dict["rh_l1"] = master_val
                    joint_dict["rh_l2"] = master_val
                    joint_dict["rh_r2"] = master_val

                    self.joint_trajectory_cmd = [
                        joint_dict.get(name, 0.0) for name in self.joint_names
                    ]
        except Exception as e:
            print("Subscriber thread exception:", e)
        finally:
            self.sub.Close()
            print("Subscriber closed")

    def publish_joint_state(self):
        # Generate timestamp
        now = datetime.now()
        stamp = Time_(
            sec=int(now.timestamp()),
            nanosec=now.microsecond * 1000
        )
        header = Header_(
            stamp=stamp,
            frame_id="base_link"
        )

        # Convert to 1D list
        positions = self.env.scene["robot"].data.joint_pos.squeeze(0).tolist()
        velocities = self.env.scene["robot"].data.joint_vel.squeeze(0).tolist()
        efforts = [0.0] * len(positions)

        # Prevent flattening 2D lists
        if isinstance(positions[0], list):
            positions = [p for sub in positions for p in sub]
        if isinstance(velocities[0], list):
            velocities = [v for sub in velocities for v in sub]

        filtered_names, filtered_positions, filtered_velocities, filtered_efforts = [], [], [], []
        for name, pos, vel, eff in zip(self.joint_names, positions, velocities, efforts):
            if name not in self.exclude_joints:
                filtered_names.append(name)
                filtered_positions.append(pos)
                filtered_velocities.append(vel)
                filtered_efforts.append(eff)

        joint_state = JointState_(
            header=header,
            name=filtered_names,
            position=filtered_positions,
            velocity=filtered_velocities,
            effort=filtered_efforts
        )

        try:
            self.pub_joint.Write(joint_state)
        except Exception as e:
            print("[Writer] write sample error. msg:", e.args)

    def publish_camera(self, cam_name: str):
        try:
            cam_data = self.env.scene[cam_name].data
            img_tensor = cam_data.output['rgb']            # get tensor
            img = img_tensor[0].cpu().numpy()             # convert to (H, W, 3) numpy array

            _, buffer = cv2.imencode('.jpg', img)
            jpeg_bytes = buffer.tobytes()

            now = datetime.now()
            stamp = Time_(
                sec=int(now.timestamp()),
                nanosec=now.microsecond * 1000
            )
            header = Header_(
                stamp=stamp,
                frame_id="camera_frame"
            )

            msg = CompressedImage_(
                header=header,
                format="jpeg",
                data=jpeg_bytes
            )
            if cam_name == "cam_wrist":
                self.pub_wrist_cam.Write(msg)
            elif cam_name == "cam_top":
                self.pub_top_cam.Write(msg)

        except Exception as e:
            print("Camera publish error:", e)

    def shutdown(self):
        self.running = False
        self.thread.join()
        # DDS cleanup
        self.sub.Close()
        self.pub_joint.Close()
        self.pub_top_cam.Close()
        self.pub_wrist_cam.Close()
        print("OMYSdk shutdown complete")

    def reset(self):
        self._reset_state = False

    def is_started(self):
        """Check if the robot control is started (B key pressed)."""
        return self._started

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
        return state

    def get_device_state(self):
        if self.joint_trajectory_cmd is None:
            return {name: 0.0 for name in self.joint_names}
        return {name: pos for name, pos in zip(self.joint_names, self.joint_trajectory_cmd)}

    def get_action(self):
        """Return action tensor for robot control (separate from observation publishing)."""
        action = self.input2action()
        if action is None:
            return self.env.action_manager.action
        if action['reset']:
            return {"reset": True}
        if not action['started']:
            # When not started, return the current robot position to maintain pose
            current_positions = self.env.scene["robot"].data.joint_pos.squeeze(0)
            # Ensure the tensor has the correct batch dimension
            if current_positions.dim() == 1:
                current_positions = current_positions.unsqueeze(0)
            return current_positions

        joint_state = action['joint_state']
        positions = [joint_state.get(name, 0.0) for name in self.joint_names]
        return torch.tensor(positions, device=self.env.device, dtype=torch.float32).unsqueeze(0)

    def publish_observations(self):
        self.publish_joint_state()
        self.publish_camera("cam_top")
        self.publish_camera("cam_wrist")

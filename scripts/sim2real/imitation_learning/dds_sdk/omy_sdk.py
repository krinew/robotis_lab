import os
import threading
import torch
import cv2
from pynput.keyboard import Listener
from collections.abc import Callable
from datetime import datetime

from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_
from robotis_dds_python.idl.sensor_msgs.msg import JointState_
from robotis_dds_python.idl.sensor_msgs.msg import CompressedImage_
from robotis_dds_python.idl.std_msgs.msg import Header_
from robotis_dds_python.idl.builtin_interfaces.msg import Time_

from robotis_dds_python.tools.topic_manager import TopicManager


class OMYSdk:
    """OMYSdk class for DDS teleoperation and publishing robot state/images."""

    def __init__(self, env, mode: str):
        self.env = env
        self.mode = mode  # 'record' or 'inference'
        self.running = True
        self.domain_id = int(os.getenv("ROS_DOMAIN_ID", 0))
        self.joint_trajectory_cmd = None
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}
        self.lock = threading.Lock()  # Protect shared state

        # Define joint names
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "rh_r1_joint"]
        self.exclude_joints = []

        # DDS Topic Manager
        topic_manager = TopicManager(domain_id=self.domain_id)

        # Subscribers
        self.joint_trajectory_reader = topic_manager.topic_reader(
            topic_name="leader/joint_trajectory",
            topic_type=JointTrajectory_
        )

        # Publishers
        self.joint_state_writer = topic_manager.topic_writer(
            topic_name="joint_states",
            topic_type=JointState_
        )
        self.top_cam_writer = topic_manager.topic_writer(
            topic_name="camera/cam_top/color/image_rect_raw/compressed",
            topic_type=CompressedImage_
        )
        self.wrist_cam_writer = topic_manager.topic_writer(
            topic_name="camera/cam_wrist/color/image_rect_raw/compressed",
            topic_type=CompressedImage_
        )

        # Start subscriber thread
        self.thread = threading.Thread(target=self._subscriber_loop, daemon=True)
        self.thread.start()

        # Keyboard listener
        self.listener = Listener(on_press=self._on_press)
        self.listener.start()

        self._keyboard_controls()

    # ----------------------
    # Keyboard controls
    # ----------------------
    def _keyboard_controls(self):
        print("\n[Control] Press keys to control the robot:")
        if self.mode == 'record':
            print("[N] New demonstration (reset simulation and start new demo)")
            print("[R] Reset simulation (observations continue)")
            print("[B] Start robot control")
        elif self.mode == 'inference':
            print("[R] Reset simulation (observations continue)")
            print("[B] Start/Resume robot control")
            print("Note: Images and joint states are always published\n")

    def _on_press(self, key):
        try:
            if self.mode == 'record':
                if key.char == 'b':
                    self._started = True
                    self._reset_state = False
                elif key.char == 'r':
                    self._started = False
                    self._reset_state = True
                    self._call_callback("R")
                elif key.char == 'n':
                    self._started = False
                    self._reset_state = True
                    self._call_callback("N")
            elif self.mode == 'inference':
                if key.char == 'b':
                    self._started = True
                    self._reset_state = False
                elif key.char == 'r':
                    self._started = False
                    self._reset_state = True
                    self._call_callback("R")
        except AttributeError:
            pass

    def _call_callback(self, key):
        if key in self._additional_callbacks:
            self._additional_callbacks[key]()

    # ----------------------
    # Subscriber loop
    # ----------------------
    def _subscriber_loop(self):
        """Continuously read joint trajectory commands from the leader."""
        try:
            while self.running:
                for msg in self.joint_trajectory_reader.take_iter():
                    if msg and msg.points:
                        joint_dict = dict(zip(msg.joint_names, msg.points[-1].positions))
                        with self.lock:
                            self.joint_trajectory_cmd = [
                                joint_dict.get(name, 0.0) for name in self.joint_names
                            ]
        except Exception as e:
            print("Subscriber thread exception:", e)
        finally:
            try:
                self.joint_trajectory_reader.Close()
            except:
                pass
            print("Subscriber closed")

    # ----------------------
    # Publishers
    # ----------------------
    def _publish_joint_states(self):
        """Publish current joint states over DDS."""
        now = datetime.now()
        stamp = Time_(sec=int(now.timestamp()), nanosec=now.microsecond * 1000)
        header = Header_(stamp=stamp, frame_id="base_link")

        positions = self.env.scene["robot"].data.joint_pos.squeeze(0).tolist()
        velocities = self.env.scene["robot"].data.joint_vel.squeeze(0).tolist()
        efforts = [0.0] * len(positions)

        # Flatten nested lists if necessary
        if isinstance(positions[0], list):
            positions = [p for sub in positions for p in sub]
        if isinstance(velocities[0], list):
            velocities = [v for sub in velocities for v in sub]

        joint_state = JointState_(
            header=header, name=list(self.joint_names),
            position=list(positions),
            velocity=list(velocities),
            effort=list(efforts))

        try:
            self.joint_state_writer.write(joint_state)
        except Exception as e:
            print("[Writer] write error:", e)

    def _publish_camera(self, cam_name: str):
        """Publish camera image as DDS compressed image."""
        try:
            cam_data = self.env.scene[cam_name].data
            img = cam_data.output['rgb'][0].cpu().numpy()  # Convert tensor to numpy

            _, buffer = cv2.imencode('.jpg', img)
            jpeg_bytes = buffer.tobytes()

            now = datetime.now()
            stamp = Time_(sec=int(now.timestamp()), nanosec=now.microsecond * 1000)
            header = Header_(stamp=stamp, frame_id="camera_frame")

            msg = CompressedImage_(header=header, format="jpeg", data=jpeg_bytes)
            if cam_name == "cam_wrist":
                self.wrist_cam_writer.write(msg)
            elif cam_name == "cam_top":
                self.top_cam_writer.write(msg)
        except Exception as e:
            print("Camera publish error:", e)

    # ----------------------
    # Action/state handling
    # ----------------------
    def _compute_action_state(self):
        """Compute current action dictionary based on keyboard input and subscriber."""
        state = {'reset': self._reset_state, 'started': self._started}
        if state['reset']:
            self._reset_state = False
            return state
        state['joint_state'] = self._get_device_state()
        return state

    def _get_device_state(self):
        """Return latest joint positions."""
        with self.lock:
            if self.joint_trajectory_cmd is None:
                return {name: 0.0 for name in self.joint_names}
            return dict(zip(self.joint_names, self.joint_trajectory_cmd))

    def get_action(self):
        """Return action tensor for robot control."""
        action = self._compute_action_state()
        if action['reset']:
            return {"reset": True}
        if not action['started']:
            return None

        joint_state = action['joint_state']
        positions = [joint_state.get(name, 0.0) for name in self.joint_names]
        return torch.tensor(positions, device=self.env.device, dtype=torch.float32).unsqueeze(0)

    def publish_observations(self):
        """Publish joint states and camera images."""
        self._publish_joint_states()
        self._publish_camera("cam_top")
        self._publish_camera("cam_wrist")

    # ----------------------
    # Utility
    # ----------------------
    def shutdown(self):
        """Stop threads and close DDS publishers/subscribers."""
        self.running = False
        self.thread.join()
        for obj in [self.joint_trajectory_reader, self.joint_state_writer,
                    self.top_cam_writer, self.wrist_cam_writer]:
            try:
                obj.Close()
            except:
                pass
        print("OMYSdk shutdown complete")

    def reset(self):
        self._reset_state = False

    def add_callback(self, key: str, func: Callable):
        """Add callback function for a specific key."""
        self._additional_callbacks[key] = func

# OMY Pick-and-Place RL Training Pipeline

This document provides a detailed, step-by-step explanation of how the OMY robot is trained for a pick-and-place task using reinforcement learning within the Isaac Lab framework.

---

## **Step-by-Step: How OMY Pick-and-Place RL Training Works**

This is the complete flow from USD asset to a trained RL policy.

### **1. USD Asset Definition (The Robot)**

**Location:** `/source/robotis_lab/robotis_lab/assets/robots/OMY.py`

- The OMY robot is defined as an `ArticulationCfg` that points to the USD file at `{ROBOTIS_LAB_ASSETS_DATA_DIR}/robots/OMY/OMY.usd`.
- The USD contains the robot's mesh, materials, articulation hierarchy (joints), and physics collision meshes.

**Key configuration:**
```python
OMY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOTIS_LAB_ASSETS_DATA_DIR}/robots/OMY/OMY.usd",
        rigid_props=...,  # Physics properties
        articulation_props=...,  # Self-collision, solver iterations
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={...}  # Default pose
    ),
    actuators={
        "DY_80": ImplicitActuatorCfg(...),  # Joint 1-2 motors
        "DY_70": ImplicitActuatorCfg(...),  # Joint 3-6 motors
        "gripper": ImplicitActuatorCfg(...), # Gripper motor
    }
)
```
*The actuator configs define PD gains (stiffness/damping) and effort/velocity limits that control how motors respond to commands.*

---

### **2. Scene Setup (Objects + Robot + Sensors)**

**Location:** `/source/robotis_lab/robotis_lab/real_world_tasks/manager_based/OMY/pick_place/joint_pos_env_cfg.py`

The environment assembles a complete scene with:

- **Robot:** `OMY_CFG` is spawned.
- **Objects (USD assets):**
  - **Table:** `OMY_TABLE_CFG` (`data/object/robotis_omy_table.usd`)
  - **Bottle:** `PLASTIC_BOTTLE_CFG` (`data/object/plastic_bottle.usd`) – the object to pick.
  - **Basket:** `PLASTIC_BASKET_CFG` (`data/object/plastic_basket.usd`) – the target container.
- **Sensors (Cameras):**
  - **Wrist camera** (`cam_wrist`): Attached to `link6` (end-effector), 848x480 RGB.
  - **Top camera** (`cam_top`): Attached to the table's `camera_link`, 848x480 RGB.
- **End-effector frame transformer:** A `FrameTransformerCfg` to track the gripper's tool-center-point (TCP).
- **Lighting:** A DomeLight for domain randomization.
- **Ground plane.**

---

### **3. Action Space (What the Policy Controls)**

**Location:** `pick_place_env_cfg.py` -> `init_action_cfg()` method

The environment supports two action modes:

1.  **Joint Position Control:** The policy outputs target joint positions for the 7 joints (6-DOF arm + 1-DOF gripper). This is used for recording demonstrations and inference.
2.  **Differential IK Control:** The policy outputs the desired end-effector pose (position + orientation), and Isaac Sim's IK solver computes the required joint positions. This is used for imitation learning.

---

### **4. Observation Space (What the Policy Sees)**

**Location:** `pick_place_env_cfg.py` -> `ObservationsCfg`

The policy observes a combination of proprioceptive state, vision, and task-specific indicators.

- **Proprioceptive state:**
  - `joint_pos` & `joint_vel`: Current positions and velocities of all 7 joints.
  - `joint_pos_target`: The target joint positions from the previous action.
  - `ee_frame_state`: The end-effector's pose relative to the robot's base.
  - `actions`: The last action taken.
- **Vision:**
  - `cam_wrist`: RGB image from the wrist camera (848x480x3).
  - `cam_top`: RGB image from the top-down camera (848x480x3).
- **Subtask indicators:**
  - `grasp_bottle`: A boolean indicating if the bottle is currently grasped.
  - `bottle_in_basket`: A boolean indicating if the bottle is inside the basket.

---

### **5. Reset and Randomization (Domain Randomization)**

**Location:** `joint_pos_env_cfg.py` -> `EventCfg`

At the start of each episode, the environment is randomized to improve generalization:

- **Robot Initialization:** The OMY arm is set to a default home pose.
- **Joint Noise:** Small Gaussian noise is added to the initial joint positions.
- **Object Pose Randomization:** The bottle and basket are spawned at random positions and orientations within a defined range.
- **Lighting Randomization:** The intensity and color of the dome light are varied.
- **Camera Pose Randomization:** Minor positional and rotational noise is applied to the top camera.

---

### **6. Termination Conditions**

**Location:** `pick_place_env_cfg.py` -> `TerminationsCfg`

An episode ends under one of the following conditions:

1.  **Success:** The bottle is successfully placed inside the basket. This is checked by `mdp.task_done`, which verifies that the bottle is within a horizontal distance threshold of the basket's center and at the correct height.
2.  **Failure:** The bottle is dropped and falls below the ground plane.
3.  **Timeout:** The episode runs for longer than the maximum allowed time (30 seconds).

---

### **7. Simulation Step Loop**

- **Physics Timestep:** `dt = 0.01` (100 Hz).
- **Decimation:** `5`. The policy runs at 20 Hz (100 Hz / 5).
- **Episode Length:** 30 seconds, which corresponds to 600 policy steps.

In each environment step, the system applies the action, steps the physics simulation 5 times, collects new observations, and checks for termination.

---

### **8. Training Entry Point**

**Location:** `scripts/reinforcement_learning/rsl_rl/train.py`

The training process is initiated with a command like:
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task RobotisLab-Real-Pick-Place-Bottle-OMY-v0 \
    --num_envs 4096 \
    --headless
```

- **Task Registration:** The `RobotisLab-Real-Pick-Place-Bottle-OMY-v0` environment is registered with `gymnasium`, pointing to the correct environment configuration class.
- **Environment Creation:** `gym.make()` creates the Isaac Lab environment, which is then wrapped for compatibility with the RSL-RL library.
- **Training Loop:** The `OnPolicyRunner` from RSL-RL is used to manage the training process. It collects rollouts from the parallel environments, computes advantages, and updates the policy and value networks using PPO.

---

### **9. Vectorized Parallel Training**

A key feature of this setup is the use of **4096 parallel environments** (`num_envs=4096`).

- Isaac Sim simulates all 4096 environments simultaneously on the GPU.
- Each environment has independent randomization (different object poses, lighting, etc.).
- Observations and actions are processed in large batches, leading to significant training speedups.

---

### **10. USD Asset Flow Summary**

1.  **USD Files:** Define the visual and physical properties of the robot and objects.
2.  **Asset Configs:** Python classes wrap the USDs, adding actuator models and initial states.
3.  **Scene Config:** Assembles all assets into a complete, interactive scene.
4.  **Environment Instantiation:** Isaac Sim loads the USDs into a stage for each of the 4096 parallel environments.
5.  **Reset & Randomization:** The `EventManager` randomizes the state of each environment at the start of an episode.
6.  **Training Loop:** The RL agent interacts with the thousands of simulated environments, learning from the batched experience to solve the task.

This pipeline demonstrates a powerful sim-to-real workflow, where declarative USD assets are brought to life in a physically accurate, parallel simulation to train robust robotic policies.

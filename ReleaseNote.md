# Robotis Lab Release Notes

0.2.0 (2025-10-28)
------------------
### OMY Sim2Real Imitation Learning Pipeline
* Folder Structure Refactor:
    * Tasks are now separated and organized into two categories:
        * real_world_tasks – for real robot execution
        * simulator_tasks – for simulation environments
* Sim2Real Pipeline Implementation:
    * Task Recording: Added functionality to record demonstrations for the OMY plastic bottle pick-and-place task in simulation.
    * Sub-task Annotation: Introduced annotation tools for splitting demonstrations into meaningful sub-tasks, improving policy learning efficiency.
    * Action Representation Conversion: Converted control commands from joint-space to IK-based end-effector pose commands for better real-world transfer.
    * Data Augmentation: Added augmentation techniques to increase dataset diversity and enhance policy generalization.
    * Dataset Conversion: Integrated data conversion to the LeRobot dataset format, enabling compatibility with LeRobot’s training framework.
* ROS 2 Integration:
    * Modified to receive the leader’s /joint_trajectory values using the robotis_dds_python library without any ROS 2 dependency.

0.1.2 (2025-07-29)
------------------
### FFW BG2 Pick-and-Place Imitation Learning Environment
Built an imitation learning environment for cylindrical rod pick-and-place using the FFW BG2 robot.

* Implemented the full pipeline:
    * Data recording
    * Sub-task annotation
    * Data augmentation
    * Training

Enabled observation input support for right_wrist_cam and head_cam.
Fixed the issue with OMY STACK task not functioning correctly.
Performed parameter tuning and code cleanup for OMY Reach task's Sim2Real code (no functional issues, just improvements).

0.1.1 (2025-07-16)
------------------
### Sim2Real Deployment Support Added
* Developed Sim2Real deployment pipeline for the OMY Reach task.
* Enabled running policies trained in Isaac Sim on real-world OMY robots.
* Provided detailed usage instructions and demonstration videos in the README.
* Refactored folder structure for source and scripts to improve maintainability.

0.1.0 (2025-07-01)
------------------
### Initial Release
* Developed as an external package for Isaac Lab
* Verified compatibility with the following environments:
    * Isaac Sim 4.5.0 and 5.0.0
    * Isaac Lab 2.1.0 and feature/isaacsim_5_0 branch
* Introduced simulation environments for reinforcement learning and imitation learning, featuring two Robotis robots:
    * OMY
    * FFW
* Enables users to conduct training and research using Robotis robots with Isaac Lab, including full support for custom tutorials in reinforcement and imitation learning scenarios.

# Hybrid Reinforcement Learning and Model Predictive Control for a 3-RPR Planar Parallel Manipulator

This repository presents a control framework for a 3-RPR planar parallel manipulator using three strategies:  
1. Model Predictive Control (MPC)  
2. Deep Reinforcement Learning (DDPG)  
3. A hybrid approach combining MPC with DDPG for improved learning stability and control performance.



## System Overview

- **Robot Type**: 3-RPR planar parallel manipulator  
- **Actuation**: 3 prismatic joints (active), with revolute joints (passive) at the base and platform  
- **Degrees of Freedom**: 3 — two translations (X, Y) and one rotation (Theta) in a 2D plane  
- **State Variables**: Position, orientation, and their velocities  
- **Control Inputs**: Actuator forces applied at the prismatic joints  
- **Goal**: Track desired end-effector trajectories with minimal error and stable control actions  



## Control Strategies

### 1. Model Predictive Control (MPC)

- Uses nonlinear optimization to compute a sequence of control inputs over a prediction horizon.  
- Minimizes a cost function based on tracking error and actuator effort.  
- Solved using `scipy.optimize.minimize`.  
- Only the first control input is applied at each step (receding horizon).  
- Includes constraints for actuator limits and dynamic feasibility.

### 2. Deep Deterministic Policy Gradient (DDPG)

- A deep reinforcement learning method for continuous action spaces.  
- Implemented using an actor-critic architecture in TensorFlow.  
- Trains a policy to map observed states to control actions (forces).  
- Uses a replay buffer to store past transitions and improve sample efficiency.  
- Soft updates ensure stable learning of target networks.

### 3. Hybrid RL + MPC (Proposed)

- Combines the structure of MPC with the learning capability of DDPG.  
- MPC-generated trajectories are added to the replay buffer.  
- DDPG trains on both exploration and expert-guided samples.  
- This improves convergence speed, sample efficiency, and overall stability.



## Modeling

### Kinematics

- **Inverse Kinematics**: Derived analytically to compute actuator lengths based on desired end-effector pose.  
- **Forward Kinematics**: Solved numerically due to closed-loop constraints.  
- **Velocity Kinematics**: The Jacobian matrix is used to relate actuator velocities to the end-effector's linear and angular velocities.

### Dynamics

- Derived using the Euler-Lagrange method.  
- Models the kinetic energy of each limb and the platform.  
- The dynamic model includes mass/inertia effects, Coriolis forces, and actuator torques.  
- Gravity is neglected due to planar horizontal configuration.  
- Dynamics are implemented symbolically and evaluated numerically for real-time use in MPC and simulation during RL training.



## Evaluation Summary

| Controller         | Convergence Speed | Tracking Accuracy | Stability        |
|--------------------|-------------------|--------------------|------------------|
| DDPG               | Slow              | Moderate           | Unstable         |
| MPC                | Fast              | High               | Stable           |
| Hybrid RL + MPC    | Fastest           | High               | Very Stable      |

- The hybrid controller showed significant improvements in training time and control smoothness.  
- MPC alone was fast and reliable but lacked adaptability.  
- Pure DDPG was unstable and slower to converge.

## Technologies Used

- Python 3.10  
- TensorFlow (for DDPG)  
- SciPy (for MPC optimization)  
- NumPy, Matplotlib (simulation and visualization)


## Future Work

- Deploy on real hardware using torque control  
- Test on 6-DOF spatial parallel manipulators  
- Use advanced RL methods like Soft Actor Critic (SAC) or Proximal Policy Optimization (PPO)  
- Apply domain randomization for sim-to-real transfer  
- Introduce online learning for system adaptation



# Hybrid Reinforcement Learning and Model Predictive Control for a 3-RPR Planar Parallel Manipulator

This repository presents an adaptive control framework for a 3-RPR planar parallel manipulator, integrating Deep Reinforcement Learning (DDPG) with Model Predictive Control (MPC). The system enables high-precision end-effector trajectory tracking in a closed-loop, nonlinear robotic mechanism with complex constraints.



## System Description

- **Manipulator Type**: 3-RPR planar parallel robot with three prismatic actuators and two passive revolute joints per leg.
- **Degrees of Freedom**: 3 (X, Y, θ)
- **Control Objective**: Track desired end-effector trajectories while satisfying physical constraints and minimizing control effort.



## Control Strategies Implemented

1. **Model Predictive Control (MPC)**
   - Finite horizon trajectory optimization using nonlinear programming.
   - Objective function penalizes pose error, actuator effort, and state deviation.
   - Constraints include actuator bounds and dynamic feasibility.
   - Solved using `scipy.optimize.minimize` in a receding horizon manner.

2. **Deep Reinforcement Learning (DDPG)**
   - Actor-Critic based deep reinforcement learning for continuous control.
   - Implemented using TensorFlow with custom actor, critic, and replay buffer.
   - Trained to minimize trajectory error using sparse reward signals.

3. **Hybrid RL + MPC (Proposed)**
   - Uses MPC-generated expert transitions to guide DDPG learning.
   - Augments replay buffer with structured demonstrations to improve convergence.
   - Combines the generalization ability of RL with the planning stability of MPC.



## Modeling and Simulation

- **Kinematics**: Full analytical derivation of forward and inverse position/velocity kinematics.
- **Dynamics**: Euler-Lagrange formulation used for deriving manipulator dynamics.
- **Simulation**: Includes end-effector trajectory tracking with noise and perturbations.
- **Reward Design**: Penalizes position error, sudden actuator inputs, and control divergence.



## Results Summary

| Strategy         | Convergence Speed | Tracking Accuracy | Stability |
|------------------|-------------------|-------------------|-----------|
| DDPG             | Low               | Moderate          | Unstable  |
| MPC              | High              | High              | Stable    |
| Hybrid RL + MPC  | Highest (~40% ↑)  | High              | Very Stable |

- The hybrid strategy showed faster learning, smoother control outputs, and reduced actuator energy usage.


## Technologies

- Python 3.10
- TensorFlow (DDPG agent)
- SciPy (nonlinear optimization)
- NumPy, Matplotlib (simulation and visualization)



## Future Work

- Extension to 3D platforms (e.g., Stewart platform).
- Hardware-in-the-loop testing with real actuators.
- Integration with advanced RL algorithms (e.g., SAC, PPO).
- Domain randomization for sim-to-real transfer.




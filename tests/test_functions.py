import os
import sys
import numpy as np
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import functions, robot_properties, symbolic_functions, fast_dynamics

def test_robot_parameters():
    
    assert robot_properties.robot_parameters() == {'rax': 0, 'ray': 0, 'rbx': 1, 'rby': 0, 'rcx': 1, 'rcy': 2, 'a1': 0.1, 'b1': 0.3, 'a2': 0.1, 'b2': 0.3, 'a3': 0.1, 'b3': 0.3, 'c12': 0.15, 'c23': 0.15, 'c31': 0.15}

def test_positional_kinematics():
    robot_pose = (0.5, 0.5, np.pi/4)
    expected_output = {
        'd1': 0.23358818077834897,
        'd2': 0.26802900867433943,
        'd3': 1.1443527812196415,
        'theta1': 0.7170019230231274,
        'theta2': 2.2436867128394082,
        'theta3': -1.9427246528694813,
        'phi12': 0.0683962403743209,
        'phi23': 0.6361065529512353,
        'phi31': 0.6337277138737338
    }
    assert np.isclose(list(functions.positional_kinematics(robot_pose, robot_properties.robot_parameters()).values()), list(expected_output.values())).all()


def test_velocity_kinematics():
    robot_state = (0.5, 0.5, math.radians(45), 1.0, 0.5, 0.1) 
    robot_params = robot_properties.robot_parameters()
    expected_output = np.array([ 1.08336825, -0.23223718, -0.81431761, -0.4659226 , -1.63712647,
        0.48665733,  0.5659226 ,  1.73712647, -0.38665733]) 
    result = functions.velocity_kinematics(robot_state, robot_params)
    assert np.isclose(list(result.values()), expected_output).all()


def test_Lagrangian():
    robot_state = (0.5, 0.5, math.radians(45), 1.0, 0.5, 0.1)
    robot_params = robot_properties.robot_parameters()
    L_func = symbolic_functions.Lagrangian_as_function_of_jointcoordinates(robot_params)
    assert np.isclose(functions.Lagrangian(robot_state, robot_params, L_func), 3.7575290669319115)

def test_fastdynamics():
    robot_dynamics = fast_dynamics.robot_dynamics_func()
    state = [1, 2, 3, 4, 5, 6]
    forces = [7, 8, 9]
    expected_output = [4, 5, 6, -381.37533588, 185.87003147, 501.19905321]
    assert np.isclose(robot_dynamics(state, forces), expected_output).all()
     

import numpy as np
from symbolic_functions import get_matrices_for_jacobian



def R(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])


def find_endeffector_vertices(robot_pose, robot_params):
    x, y, theta = robot_pose
    
    pe = np.array([x,y])    

    _, _, _, _, _, _, _, _, _, _, _, _, c12, c23, c31 = robot_params.values()
    phi = np.arccos((c12**2+c31**2-c23**2)/(2*c12*c31))
    xc = (c12+c31*np.cos(phi))/3
    yc = (c31*np.sin(phi))/3
    pc = np.array([xc,yc])
    
    p1 = np.array([0,0])
    p2 = np.array([c12,0])
    p3 = np.array([c31*np.cos(phi),c31*np.sin(phi)])
    
    P1bar = p1 - pc
    P2bar = p2 - pc
    P3bar = p3 - pc
    
    P1_transformed = R(theta) @ P1bar + pe
    P2_transformed = R(theta) @ P2bar + pe
    P3_transformed = R(theta) @ P3bar + pe
    
    return (P1_transformed, P2_transformed, P3_transformed)
    


def positional_kinematics(robot_pose, robot_params):
    
    rax, ray, rbx, rby, rcx, rcy, a1, a2, a3, b1, b2, b3, _, _, _ = robot_params.values()
    
    x1y1, x2y2, x3y3 = find_endeffector_vertices(robot_pose, robot_params)
    
    x1, y1 = x1y1
    x2, y2 = x2y2
    x3, y3 = x3y3

    d1 = np.sqrt((x1-rax)**2+(y1-ray)**2) - (a1+b1)
    d2 = np.sqrt((x2-rbx)**2+(y2-rby)**2) - (a2+b2)
    d3 = np.sqrt((x3-rcx)**2+(y3-rcy)**2) - (a3+b3)
    theta1 = np.arctan2(y1-ray, x1-rax)
    theta2 = np.arctan2(y2-rby, x2-rbx)
    theta3 = np.arctan2(y3-rcy, x3-rcx)
    phi12 = np.arctan2(y2-y1,x2-x1) - theta1
    phi23 = np.arctan2(y3-y2,x3-x2) - theta2
    phi31 = np.arctan2(y1-y3,x1-x3) - theta3
    dictionary = {
        'd1': d1,
        'd2': d2,
        'd3': d3,
        'theta1': theta1,
        'theta2': theta2,
        'theta3': theta3,
        'phi12': phi12,
        'phi23': phi23,
        'phi31': phi31,
    }
    return dictionary

def jacobian(robot_state, robot_params, other_matrices = False):
    ee_position = robot_state[:3]
    
    positions_dictionary = positional_kinematics(ee_position, robot_params)
    Ja, Jp, Aa, Ap = get_matrices_for_jacobian()
    Jav = Ja(robot_params.values(), positions_dictionary.values())
    Jpv = Jp(robot_params.values(), positions_dictionary.values())
    Aav = Aa(robot_params.values(), positions_dictionary.values())
    Apv = Ap(robot_params.values(), positions_dictionary.values())
    J = Jav - Jpv @ np.linalg.inv(Apv) @ Aav

    if other_matrices == True:
        return J, [Jav, Jpv, Aav, Apv]
    elif other_matrices == False:
        return J

def velocity_kinematics(robot_state, robot_params):
    ee_velocity = robot_state[3:]

    J, other_matrices = jacobian(robot_state, robot_params, other_matrices = True)
    _, _, Aav, Apv = other_matrices
    
    qda = np.linalg.inv(J) @ ee_velocity
    qdp = -np.linalg.inv(Apv)@Aav@qda

    dictionary = {
        'd1_dot': qda[0],
        'd2_dot': qda[1],
        'd3_dot': qda[2],
        'theta1_dot': qdp[0],
        'theta2_dot': qdp[1],
        'theta3_dot': qdp[2],
        'phi12_dot': qdp[3],
        'phi23_dot': qdp[4],
        'phi31_dot': qdp[5],
    }
    return dictionary

def Lagrangian(robot_state, robot_params, L_func):
    x, y, theta, _, _, _ = robot_state
    robot_pose = np.array([x, y, theta])
    
    joint_values = positional_kinematics(robot_pose, robot_params)
    joint_velocities = velocity_kinematics(robot_state, robot_params)

    return L_func(joint_values.values(), joint_velocities.values())

def derivative(func, x, index, h = 0.0001):
    x = np.array(x, float)
    xph = x.copy()
    xmh = x.copy()
    xph[index] += h
    xmh[index] -= h
    fxph = func(xph)
    fxmh = func(xmh)
    return ((fxph-fxmh)/(2*h))

def doublederivative(func, x, indices, h = 0.0001, k = 0.0001):
    index1, index2 = indices
    x = np.array(x, float)
    x1 = x.copy()
    x2 = x.copy()
    x3 = x.copy()
    x4 = x.copy()

    x1[index1] += h
    x1[index2] += k

    x2[index1] -= h
    x2[index2] -= k

    x3[index1] += h
    x3[index2] -= k

    x4[index1] -= h
    x4[index2] += k

    f_x1 = func(x1)
    f_x2 = func(x2)
    f_x3 = func(x3)
    f_x4 = func(x4)

    return ((f_x1+f_x2-f_x3-f_x4)/(4*h*k))

def statespace(s, a, robot_params, L):
    xk = s
    Fk = a
    ddot_vals = xk[3:6]

    
    J = jacobian(s, robot_params)

    A = np.array([[doublederivative(L, xk, (j,i), h=0.001, k=0.001) for j in range(3)] for i in range(3,6)])
    B = np.array([[doublederivative(L, xk, (j,i), h=0.001, k=0.001) for j in range(3,6)] for i in range(3,6)])
    C = np.array([derivative(L, xk, i, h=0.001) for i in range(3)])

    try:
        sd2 = np.linalg.inv(B) @ (np.linalg.inv(J.T)@Fk + C - (A @ np.array(ddot_vals)))
    except:
        sd2 = np.linalg.pinv(B) @ (np.linalg.pinv(J.T)@Fk + C - (A @ np.array(ddot_vals)))
    sd1 = xk[3:6]
    sd = np.r_[sd1,sd2]
    return sd

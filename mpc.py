import numpy as np
from functions import statespace, Lagrangian
from robot_properties import robot_parameters
import math
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from symbolic_functions import Lagrangian_as_function_of_jointcoordinates


robot_state = (0.01, 0.02, math.radians(15), 0, 0, 0)
target_state = (0,0,0,0,0,0)
robot_params = robot_parameters()

L_func = Lagrangian_as_function_of_jointcoordinates(robot_params)
L = lambda state: Lagrangian(state, robot_params, L_func)

def robot_dynamics(x,u):
  return statespace(x, u, robot_params, L)

max_steps = 100

horizon = 10
N = horizon 
dt = 0.1 
Q = np.diag([1, 1, 1, 0.1, 0.1, 0.1]) 
R = np.diag([0.1, 0.1, 0.1]) 


x0 = np.array(robot_state)
xd = np.array(target_state)

def cost_function(u, x, xd):
  cost = 0
  x_pred = np.zeros((N+1, len(xd)))
  x_pred[0,:] = x
  for i in range(N):
    x_pred[i+1,:] = x_pred[i,:] + dt*robot_dynamics(x_pred[i,:], u[i*len(R):(i+1)*len(R)])
    cost += (x_pred[i+1,:] - xd).T @ Q @ (x_pred[i+1,:] - xd) + u[i*len(R):(i+1)*len(R)].T @ R @ u[i*len(R):(i+1)*len(R)]
  return cost


def mpc_solver(x, xd):
  u0 = np.random.uniform(-5, 5, size=N*len(R))
  bounds = [(-5, 5) for _ in range(N*len(R))]
  result = minimize(cost_function, u0, args=(x, xd), bounds=bounds)
  u_opt = result.x

  return u_opt


x = x0
x_history = [x0]
u_history = []
for i in range(max_steps):
  u_opt = mpc_solver(x, xd)

  u = u_opt[:len(R)]
  u_history.append(u)

  x_dot = robot_dynamics(x, u)
  x = x + dt*x_dot
  x_history.append(x)
  print(f'Step {i+1} completed out of {len(max_steps)}...')
  

t_vals = np.arange(0, (len(x_history)-1)*dt, dt)
x_history = np.array(x_history)
u_history = np.array(u_history)

plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(t_vals, x_history[:-1, 0], label="x")
plt.plot(t_vals, x_history[:-1, 1], label="y")
plt.plot(t_vals, x_history[:-1, 2], label="y")
plt.xlabel("Time (s)")
plt.ylabel("End-effector state (m and rad)")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t_vals, x_history[:-1, 3], label=r"\dot{x}")
plt.plot(t_vals, x_history[:-1, 4], label=r"\dot{y}")
plt.plot(t_vals, x_history[:-1, 5], label=r"\dot{\theta}")
plt.xlabel("Time (s)")
plt.ylabel("Joint Velocity (rad/s)")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t_vals, u_history[:, 0], label="F_1")
plt.plot(t_vals, u_history[:, 1], label="F_2")
plt.plot(t_vals, u_history[:, 2], label="F_3")
plt.xlabel("Time (s)")
plt.ylabel("Control Input (N)")
plt.legend()

plt.tight_layout()
plt.savefig(f'result.png', bbox_inches='tight', transparent=True, dpi=1200)
plt.show()

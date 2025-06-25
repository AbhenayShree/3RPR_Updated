import numpy as np
# import math
from scipy.optimize import minimize
# from matplotlib import pyplot as plt
from fast_dynamics import robot_dynamics_func

# robot_state = (0.5, 0.5, math.radians(45), 1.0, 0.5, 0.1)
# target_state = (0,0,0,0,0,0)

robot_dynamics = robot_dynamics_func()

# max_steps = 100

# horizon = 10
# N = horizon 
# dt = 0.1 
# Q = np.diag([1, 1, 1, 0.1, 0.1, 0.1]) 
# R = np.diag([0.1, 0.1, 0.1]) 


# x0 = np.array(robot_state)
# xd = np.array(target_state)

# def cost_function(u, x, xd, N, dt, Q, R):
#   cost = 0
#   x_pred = np.zeros((N+1, len(xd)))
#   x_pred[0,:] = x
#   for i in range(N):
#     x_pred[i+1,:] = x_pred[i,:] + dt*robot_dynamics(x_pred[i,:], u[i*len(R):(i+1)*len(R)])
#     cost += (x_pred[i+1,:] - xd).T @ Q @ (x_pred[i+1,:] - xd) + u[i*len(R):(i+1)*len(R)].T @ R @ u[i*len(R):(i+1)*len(R)]
#   return cost


# def mpc_solver(x, xd, N, dt, Q, R):
#   u0 = np.random.uniform(-5, 5, size=N*len(R))
#   bounds = [(-5, 5) for _ in range(N*len(R))]
#   result = minimize(cost_function, u0, args=(x, xd), bounds=bounds)
#   u_opt = result.x

#   return u_opt


# x = x0
# x_history = [x0]
# u_history = []
# for i in range(max_steps):
#   u_opt = mpc_solver(x, xd, N, dt, Q, R)

#   u = u_opt[:len(R)]
#   u_history.append(u)

#   x_dot = robot_dynamics(x, u)
#   x = x + dt*x_dot
#   x_history.append(x)
#   print(f'Step {i+1} completed out of {max_steps}...')
  

# t_vals = np.arange(0, (len(x_history)-1)*dt, dt)
# x_history = np.array(x_history)
# u_history = np.array(u_history)

# plt.figure(figsize=(10, 6))
# plt.subplot(2, 2, 1)
# plt.plot(t_vals, x_history[:-1, 0], label=r"$x$")
# plt.plot(t_vals, x_history[:-1, 1], label=r"$y$")
# plt.plot(t_vals, x_history[:-1, 2], label=r"$\theta$")
# plt.xlabel("Time (s)")
# plt.ylabel("End-effector state (m and rad)")
# plt.legend()

# plt.subplot(2, 2, 2)
# plt.plot(t_vals, x_history[:-1, 3], label=r"$\dot{x}$")
# plt.plot(t_vals, x_history[:-1, 4], label=r"$\dot{y}$")
# plt.plot(t_vals, x_history[:-1, 5], label=r"$\dot{\theta}$")
# plt.xlabel("Time (s)")
# plt.ylabel("Joint Velocity (rad/s)")
# plt.legend()

# plt.subplot(2, 2, 3)
# plt.plot(t_vals, u_history[:, 0], label=r"$F_1$")
# plt.plot(t_vals, u_history[:, 1], label=r"$F_2$")
# plt.plot(t_vals, u_history[:, 2], label=r"$F_3$")
# plt.xlabel("Time (s)")
# plt.ylabel("Control Input (N)")
# plt.legend()

# plt.tight_layout()
# plt.savefig(f'result.png', bbox_inches='tight', transparent=True, dpi=1200)
# plt.show()


def cost_function(u, x, xd, N, dt, Q, R):
  cost = 0
  x_pred = np.zeros((N+1, len(xd)))
  x_pred[0,:] = x
  for i in range(N):
    x_pred[i+1,:] = x_pred[i,:] + dt*robot_dynamics(x_pred[i,:], u[i*len(R):(i+1)*len(R)])
    cost += (x_pred[i+1,:] - xd).T @ Q @ (x_pred[i+1,:] - xd) + u[i*len(R):(i+1)*len(R)].T @ R @ u[i*len(R):(i+1)*len(R)]
  return cost

def mpc_solver(x, xd, N, dt, Q, R, u0):
  bounds = [(-5, 5) for _ in range(N*len(R))]
  result = minimize(cost_function, u0, args=(x, xd, N, dt, Q, R), bounds=bounds)
  u_opt = result.x

  return u_opt

def mpc(x0, xd, max_steps, N, dt, Q, R):

  x = x0
  x_history = []
  u_history = []
  u0 = np.zeros(N*len(R))
  for _ in range(max_steps):
    u_opt = mpc_solver(x, xd, N, dt, Q, R, u0)

    u = u_opt[:len(R)]
    u_history.append(u)
    u0 = u_opt.copy()

    x_dot = robot_dynamics(x, u)
    x = x + dt*x_dot
    x_history.append(x)

  t = np.arange(0, (len(x_history))*dt, dt)
  x_history = np.array(x_history)
  u_history = np.array(u_history)

  return x_history, u_history, t
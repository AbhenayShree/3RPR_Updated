import sympy
import numpy as np

from robot_properties import robot_parameters, inertial_and_geometrical_parameters as inertial_and_geometrical_parameters_func

robot_params = robot_parameters()

theta1 = sympy.symbols(r'\theta_1')
theta2 = sympy.symbols(r'\theta_2')
theta3 = sympy.symbols(r'\theta_3')

phi12 = sympy.symbols(r'\phi_{12}')
phi23 = sympy.symbols(r'\phi_{23}')
phi31 = sympy.symbols(r'\phi_{31}')

a1 = sympy.symbols(r'a_1')
a2 = sympy.symbols(r'a_2')
a3 = sympy.symbols(r'a_3')

b1 = sympy.symbols(r'b_1')
b2 = sympy.symbols(r'b_2')
b3 = sympy.symbols(r'b_3')

d1 = sympy.symbols(r'd_1')
d2 = sympy.symbols(r'd_2')
d3 = sympy.symbols(r'd_3')

c12 = sympy.symbols(r'c_{12}')
c23 = sympy.symbols(r'c_{23}')
c31 = sympy.symbols(r'c_{31}')

rax = sympy.symbols(r'r_{Ax}')
ray = sympy.symbols(r'r_{Ay}')
rbx = sympy.symbols(r'r_{Bx}')
rby = sympy.symbols(r'r_{By}')
rcx = sympy.symbols(r'r_{Cx}')
rcy = sympy.symbols(r'r_{Cy}')

ra = sympy.Matrix([rax,ray])
rb = sympy.Matrix([rbx,rby])
rc = sympy.Matrix([rcx,rcy])

r1g = sympy.symbols(r'r_{1g}')

t = sympy.symbols('t')

theta1 = sympy.Function(r'\theta_1')(t)
theta2 = sympy.Function(r'\theta_2')(t)
theta3 = sympy.Function(r'\theta_3')(t)

phi12 = sympy.Function(r'\phi_{12}')(t)
phi23 = sympy.Function(r'\phi_{23}')(t)
phi31 = sympy.Function(r'\phi_{31}')(t)

d1 = sympy.Function(r'd_1')(t)
d2 = sympy.Function(r'd_2')(t)
d3 = sympy.Function(r'd_3')(t)


la1 = sympy.symbols(r'l_{a1}')
lb1 = sympy.symbols(r'l_{b1}')

la2 = sympy.symbols(r'l_{a2}')
lb2 = sympy.symbols(r'l_{b2}')

la3 = sympy.symbols(r'l_{a3}')
lb3 = sympy.symbols(r'l_{b3}')

ma1 = sympy.symbols(r'm_{a1}')
ma2 = sympy.symbols(r'm_{a2}')
ma3 = sympy.symbols(r'm_{a3}')

mb1 = sympy.symbols(r'm_{b1}')
mb2 = sympy.symbols(r'm_{b2}')
mb3 = sympy.symbols(r'm_{b3}')

Ia1 = sympy.symbols(r'I_{a1}')
Ia2 = sympy.symbols(r'I_{a2}')
Ia3 = sympy.symbols(r'I_{a3}')

Ib1 = sympy.symbols(r'I_{b1}')
Ib2 = sympy.symbols(r'I_{b2}')
Ib3 = sympy.symbols(r'I_{b3}')

mg = sympy.symbols(r'm_{G}')
Ig = sympy.symbols(r'I_{G}')


eitheta = lambda theta: sympy.Matrix([sympy.cos(theta),sympy.sin(theta)])

k = sympy.Matrix([
    sympy.cos(theta1), sympy.sin(theta1),
    sympy.cos(theta2), sympy.sin(theta2),
    sympy.cos(theta3), sympy.sin(theta3),
    sympy.cos(phi12), sympy.sin(phi12),
    sympy.cos(phi23), sympy.sin(phi23),
    sympy.cos(phi31), sympy.sin(phi31),
])

f1 = (a1+d1+b1)*eitheta(theta1) + c12*eitheta(theta1+phi12) - (a2+d2+b2)*eitheta(theta2) - (rb-ra)
f2 = (a2+d2+b2)*eitheta(theta2) + c23*eitheta(theta2+phi23) - (a3+d3+b3)*eitheta(theta3) - (rc-rb)
f3 = (a3+d3+b3)*eitheta(theta3) + c31*eitheta(theta3+phi31) - (a1+d1+b1)*eitheta(theta1) - (ra-rc)

f = sympy.Matrix([f1,f2,f3])

vars = sympy.Matrix([d1, d2, d3, theta1, theta2, theta3, phi12, phi23, phi31])
varsdot = sympy.diff(vars,t)

# rax_val = 0
# ray_val = 0
# rbx_val = 1
# rby_val = 0
# rcx_val = 1
# rcy_val = 2


# a1_val = 0.1
# b1_val = 0.3
# a2_val = 0.1
# b2_val = 0.3
# a3_val = 0.1
# b3_val = 0.3

# c12_val = 0.15
# c23_val = 0.15
# c31_val = 0.15

subs_vals_keys = [rax, ray, rbx, rby, rcx, rcy, a1, a2, a3, b1, b2, b3, c12, c23, c31]
# subs_vals_values = [rax_val, ray_val, rbx_val, rby_val, rcx_val, rcy_val, a1_val, a2_val, a3_val, b1_val, b2_val, b3_val, c12_val, c23_val, c31_val]
subs_vals_values = list(robot_params.values())

subs_vals = dict(zip(subs_vals_keys, subs_vals_values))



v = sympy.diff((a1+d1+b1)*eitheta(theta1)+r1g*eitheta(theta1+phi12),t)
vomega = sympy.Matrix([v, sympy.diff(theta1+phi12)])

J_full = vomega.jacobian(varsdot)
Ja = J_full[:,:3]
Jp = J_full[:,3:]

A_full = sympy.diff(f,t).jacobian(varsdot)
Aa = A_full[:,:3]
Ap = A_full[:,3:]

parametersf = sympy.Matrix(list(subs_vals.keys()))
parametersv = np.array(list(subs_vals.values()), dtype=float)

Ja_func = sympy.lambdify([parametersf, vars],Ja)
Jp_func = sympy.lambdify([parametersf, vars, r1g],Jp)
Aa_func = sympy.lambdify([parametersf, vars],Aa)
Ap_func = sympy.lambdify([parametersf, vars],Ap)



# -------------------------------


def inverse_kinematics(end_effector_state, dimensional_parameters_values):
    x, y, theta, xdot, ydot, thetadot = end_effector_state
    v_e = np.array([xdot, ydot, thetadot])
    rax_val, ray_val, rbx_val, rby_val, rcx_val, rcy_val, a1_val, a2_val, a3_val, b1_val, b2_val, b3_val, c12_val, c23_val, c31_val = dimensional_parameters_values

    phi1_val = sympy.acos((c12_val**2+c31_val**2-c23_val**2)/(2*c12_val*c31_val))
    (xb1, yb1) = (0, 0)
    (xb2, yb2) = (c12_val, 0)
    (xb3, yb3) = (c31_val*sympy.cos(phi1_val), c31_val*sympy.sin(phi1_val))
    xbc = (xb1+xb2+xb3)/3
    ybc = (yb1+yb2+yb3)/3
    (xl1, yl1) = (xb1 - xbc, yb1 - ybc)
    (xl2, yl2) = (xb2 - xbc, yb2 - ybc)
    (xl3, yl3) = (xb3 - xbc, yb3 - ybc)
    T = np.array([[np.cos(theta), -np.sin(theta), x], [np.sin(theta), np.cos(theta), y], [0, 0, 1]])
    (x1, y1, _) = T @ np.array([xl1, yl1, 1], dtype=float)
    (x2, y2, _) = T @ np.array([xl2, yl2, 1], dtype=float)
    (x3, y3, _) = T @ np.array([xl3, yl3, 1], dtype=float)
    d1_val = np.sqrt((x1 - rax_val)**2 + (y1 - ray_val)**2) - a1_val - b1_val
    d2_val = np.sqrt((x2 - rbx_val)**2 + (y2 - rby_val)**2) - a2_val - b2_val
    d3_val = np.sqrt((x3 - rcx_val)**2 + (y3 - rcy_val)**2) - a3_val - b3_val
    theta1_val = np.arctan2(y1 - ray_val, x1 - rax_val)
    theta2_val = np.arctan2(y2 - rby_val, x2 - rbx_val)
    theta3_val = np.arctan2(y3 - rcy_val, x3 - rcx_val)
    phi12_val = np.arctan2(y2-y1,x2-x1) - theta1_val
    phi23_val = np.arctan2(y3-y2,x3-x2) - theta2_val
    phi31_val = np.arctan2(y1-y3,x1-x3) - theta3_val
    
    vars_vals = np.array([d1_val, d2_val, d3_val, theta1_val, theta2_val, theta3_val, phi12_val, phi23_val, phi31_val])

    r1g_value = np.sqrt((x1 - x)**2 + (y1 - y)**2)
    Jav = Ja_func(dimensional_parameters_values, vars_vals)
    Jpv = Jp_func(dimensional_parameters_values, vars_vals, r1g_value)
    Aav = Aa_func(dimensional_parameters_values, vars_vals)
    Apv = Ap_func(dimensional_parameters_values, vars_vals)
    try:
        J = Jav - Jpv @ np.linalg.inv(Apv) @ Aav
    except:
        J = Jav - Jpv @ np.linalg.pinv(Apv) @ Aav
    try:
        qda = np.linalg.inv(J)@v_e
    except:
        qda = np.linalg.pinv(J)@v_e
    d1dot_val, d2dot_val, d3dot_val = qda
    try:
        qdp = -np.linalg.inv(Apv)@Aav@qda
    except:
        qdp = -np.linalg.pinv(Apv)@Aav@qda
    theta1dot_val, theta2dot_val, theta3dot_val, phi12dot_val, phi23dot_val, phi31dot_val = qdp
    varsdot_vals = np.array([d1dot_val, d2dot_val, d3dot_val, theta1dot_val, theta2dot_val, theta3dot_val, phi12dot_val, phi23dot_val, phi31dot_val])



    return vars_vals,varsdot_vals, J

ma1 = sympy.symbols(r'm_{a1}')
ma2 = sympy.symbols(r'm_{a2}')
ma3 = sympy.symbols(r'm_{a3}')

mb1 = sympy.symbols(r'm_{b1}')
mb2 = sympy.symbols(r'm_{b2}')
mb3 = sympy.symbols(r'm_{b3}')

Ia1 = sympy.symbols(r'I_{a1}')
Ia2 = sympy.symbols(r'I_{a2}')
Ia3 = sympy.symbols(r'I_{a3}')

Ib1 = sympy.symbols(r'I_{b1}')
Ib2 = sympy.symbols(r'I_{b2}')
Ib3 = sympy.symbols(r'I_{b3}')

mg = sympy.symbols(r'm_{G}')
Ig = sympy.symbols(r'I_{G}')

a1 = sympy.symbols(r'a_1')
a2 = sympy.symbols(r'a_2')
a3 = sympy.symbols(r'a_3')

b1 = sympy.symbols(r'b_1')
b2 = sympy.symbols(r'b_2')
b3 = sympy.symbols(r'b_3')

c12 = sympy.symbols(r'c_{12}')
c23 = sympy.symbols(r'c_{23}')
c31 = sympy.symbols(r'c_{31}')

la1 = sympy.symbols(r'l_{a1}')
lb1 = sympy.symbols(r'l_{b1}')

la2 = sympy.symbols(r'l_{a2}')
lb2 = sympy.symbols(r'l_{b2}')

la3 = sympy.symbols(r'l_{a3}')
lb3 = sympy.symbols(r'l_{b3}')

t = sympy.symbols(r't')

theta1 = sympy.Function(r'\theta_1')(t)
theta2 = sympy.Function(r'\theta_2')(t)
theta3 = sympy.Function(r'\theta_3')(t)

d1 = sympy.Function(r'd_1')(t)
d2 = sympy.Function(r'd_2')(t)
d3 = sympy.Function(r'd_3')(t)

phi12 = sympy.Function(r'\phi_{12}')(t)
phi23 = sympy.Function(r'\phi_{23}')(t)
phi31 = sympy.Function(r'\phi_{31}')(t)


rax = sympy.symbols(r'r_{Ax}')
rbx = sympy.symbols(r'r_{Bx}')
rcx = sympy.symbols(r'r_{Cx}')

ray = sympy.symbols(r'r_{Ay}')
rby = sympy.symbols(r'r_{By}')
rcy = sympy.symbols(r'r_{Cy}')

ra = sympy.Matrix([rax,ray,0])
rb = sympy.Matrix([rbx,rby,0])
rc = sympy.Matrix([rcx,rcy,0])

omega1 = sympy.diff(sympy.Matrix([0,0,theta1]),t)
omega2 = sympy.diff(sympy.Matrix([0,0,theta2]),t)
omega3 = sympy.diff(sympy.Matrix([0,0,theta3]),t)

omega12 = sympy.diff(sympy.Matrix([0,0,phi12]),t)
omega23 = sympy.diff(sympy.Matrix([0,0,phi23]),t)
omega31 = sympy.diff(sympy.Matrix([0,0,phi31]),t)



uvec = lambda th: sympy.Matrix([sympy.cos(th), sympy.sin(th), 0])

rla1 = la1*uvec(theta1)
rla2 = la2*uvec(theta2)
rla3 = la3*uvec(theta3)

rlb1 = (a1+d1+lb1)*uvec(theta1)
rlb2 = (a2+d2+lb2)*uvec(theta2)
rlb3 = (a3+d3+lb3)*uvec(theta3)

vla1 = sympy.diff(rla1,t)
vla2 = sympy.diff(rla2,t)
vla3 = sympy.diff(rla3,t)

vlb1 = sympy.diff(rlb1,t)
vlb2 = sympy.diff(rlb2,t)
vlb3 = sympy.diff(rlb3,t)

rd = (a1+d1+b1)*uvec(theta1)
re = (a2+d2+b2)*uvec(theta2)
rf = (a3+d3+b3)*uvec(theta3)

vd = sympy.diff(rd,t)
ve = sympy.diff(re,t)
vf = sympy.diff(rf,t)

rg = (rd+re+rf)/3

vg = vd + omega12.cross(rg-rd)
omegag = omega1 + omega12

Ta1 = sympy.Rational(1,2)*ma1*(vla1.dot(vla1)) + sympy.Rational(1,2)*Ia1*(omega1.dot(omega1))
Ta2 = sympy.Rational(1,2)*ma2*(vla2.dot(vla2)) + sympy.Rational(1,2)*Ia2*(omega2.dot(omega2))
Ta3 = sympy.Rational(1,2)*ma3*(vla3.dot(vla3)) + sympy.Rational(1,2)*Ia3*(omega3.dot(omega3))

Tb1 = sympy.Rational(1,2)*mb1*(vlb1.dot(vlb1)) + sympy.Rational(1,2)*Ib1*(omega1.dot(omega1))
Tb2 = sympy.Rational(1,2)*mb2*(vlb2.dot(vlb2)) + sympy.Rational(1,2)*Ib2*(omega2.dot(omega2))
Tb3 = sympy.Rational(1,2)*mb3*(vlb3.dot(vlb3)) + sympy.Rational(1,2)*Ib3*(omega3.dot(omega3))

Tg = sympy.Rational(1,2)*mg*(vg.dot(vg)) + sympy.Rational(1,2)*Ig*(omegag.dot(omegag))

T = Ta1+Tb1+Ta2+Tb2+Ta3+Tb3+Tg
V = 0

L = T-V

parameters_keys = [rax, ray, rbx, rby, rcx, rcy, a1, a2, a3, b1, b2, b3, c12, c23, c31]
parameters_values = list(robot_params.values())
parameters = dict(zip(parameters_keys,parameters_values))
# parameters = {
#     rax: rax_val,
#     ray: ray_val,
#     rbx: rbx_val,
#     rby: rby_val,
#     rcx: rcx_val,
#     rcy: rcy_val,
#     a1: a1_val,
#     a2: a2_val,
#     a3: a3_val,
#     b1: b1_val,
#     b2: b2_val,
#     b3: b3_val,
#     c12: c12_val,
#     c23: c23_val,
#     c31: c31_val
# }


la1v = 0.75*parameters[a1]
la2v = 0.75*parameters[a2]
la3v = 0.75*parameters[a3]

lb1v = 0.5*parameters[b1]
lb2v = 0.5*parameters[b2]
lb3v = 0.5*parameters[b3]

ma1v = 1
ma2v = 1
ma3v = 1

mb1v = 1
mb2v = 1
mb3v = 1

mgv = 2

Ia1v = ma1v*la1v**2
Ia2v = ma2v*la2v**2
Ia3v = ma3v*la3v**2

Ib1v = mb1v*lb1v**2
Ib2v = mb2v*lb2v**2
Ib3v = mb3v*lb3v**2

Igv = mg*0.5*parameters[c12]**2

inertial_and_geometrical_parameters_keys = [la1,la2,la3,lb1,lb2,lb3,ma1,ma2,ma3,mb1,mb2,mb3,mg,Ia1,Ia2,Ia3,Ib1,Ib2,Ib3,Ig]
inertial_and_geometrical_parameters_dict = inertial_and_geometrical_parameters_func(robot_params)
inertial_and_geometrical_parameters_values = inertial_and_geometrical_parameters_dict.values()
inertial_and_geometrical_parameters = dict(zip(inertial_and_geometrical_parameters_keys,inertial_and_geometrical_parameters_values))
# inertial_and_geometrical_parameters = {
#     la1: la1v,
#     la2: la2v,
#     la3: la3v,
#     lb1: lb1v,
#     lb2: lb2v,
#     lb3: lb3v,
#     ma1: ma1v,
#     ma2: ma2v,
#     ma3: ma3v,
#     mb1: mb1v,
#     mb2: mb2v,
#     mb3: mb3v,
#     mg: mgv,
#     Ia1: Ia1v,
#     Ia2: Ia2v,
#     Ia3: Ia3v,
#     Ib1: Ib1v,
#     Ib2: Ib2v,
#     Ib3: Ib3v,
#     Ig: Igv
# }

active_joint_parameters = sympy.Matrix(vars[:3])
passive_joint_parameters = sympy.Matrix(vars[3:])

passive_joint_derivatives = sympy.diff(passive_joint_parameters,t)
active_joint_derivatives = sympy.diff(active_joint_parameters,t)

thd1 =  sympy.symbols(r'\dot{\theta}_1')
thd2 =  sympy.symbols(r'\dot{\theta}_2')
thd3 =  sympy.symbols(r'\dot{\theta}_3')
ph12d =  sympy.symbols(r'\dot{\phi}_{12}')
ph23d =  sympy.symbols(r'\dot{\phi}_{23}')
ph31d =  sympy.symbols(r'\dot{\phi}_{31}')

d1d = sympy.symbols(r'\dot{d}_1')
d2d = sympy.symbols(r'\dot{d}_2')
d3d = sympy.symbols(r'\dot{d}_3')

der_variables_list_2 = sympy.Matrix([thd1, thd2, thd3, ph12d, ph23d, ph31d])
der_variables_list_1 = sympy.Matrix([d1d, d2d, d3d])

derivative_variables_2 = dict(zip(passive_joint_derivatives, der_variables_list_2))
derivative_variables_1 = dict(zip(active_joint_derivatives, der_variables_list_1))

varsdot_f = varsdot.subs(derivative_variables_1).subs(derivative_variables_2)
Lf = sympy.lambdify([vars,varsdot_f], L.subs(parameters).subs(inertial_and_geometrical_parameters).subs(derivative_variables_1).subs(derivative_variables_2))



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

def L_func(x):
    vars_vals, varsdot_vals, _ = inverse_kinematics(x, parametersv)
    return Lf(vars_vals, varsdot_vals)

def statespace(s, a, params):
    xk = s
    Fk = a
    ddot_vals = xk[3:6]

    _, _, J = inverse_kinematics(s, params)


    A = np.array([[doublederivative(L_func, xk, (j,i), h=0.001, k=0.001) for j in range(3)] for i in range(3,6)])
    B = np.array([[doublederivative(L_func, xk, (j,i), h=0.001, k=0.001) for j in range(3,6)] for i in range(3,6)])
    C = np.array([derivative(L_func, xk, i, h=0.001) for i in range(3)])

    try:
        sd2 = np.linalg.inv(B) @ (np.linalg.inv(J.T)@Fk + C - (A @ np.array(ddot_vals)))
    except:
        sd2 = np.linalg.pinv(B) @ (np.linalg.pinv(J.T)@Fk + C - (A @ np.array(ddot_vals)))
    sd1 = xk[3:6]
    sd = np.r_[sd1,sd2]
    return sd

def robot_dynamics(x,u):
    return statespace(x, u, parametersv)

def robot_dynamics_func():
    return robot_dynamics
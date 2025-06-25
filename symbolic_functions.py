import sympy
from robot_properties import inertial_and_geometrical_parameters

def get_matrices_for_jacobian():

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



    subs_vals_keys = [rax, ray, rbx, rby, rcx, rcy, a1, a2, a3, b1, b2, b3, c12, c23, c31]

    phi1 = sympy.acos((c12**2+c31**2-c23**2)/(2*c12*c31))
    r1g = sympy.sqrt((c12**2+c31**2+2*c12*c31*sympy.cos(phi1))/3)

    v = sympy.diff((a1+d1+b1)*eitheta(theta1)+r1g*eitheta(theta1+phi12),t)
    vomega = sympy.Matrix([v, sympy.diff(theta1+phi12)])

    J_full = vomega.jacobian(varsdot)
    Ja = J_full[:,:3]
    Jp = J_full[:,3:]

    A_full = sympy.diff(f,t).jacobian(varsdot)
    Aa = A_full[:,:3]
    Ap = A_full[:,3:]

    parametersf = subs_vals_keys    

    Ja_func = sympy.lambdify([parametersf, vars],Ja, modules=['numpy'])
    Jp_func = sympy.lambdify([parametersf, vars],Jp, modules=['numpy'])
    Aa_func = sympy.lambdify([parametersf, vars],Aa, modules=['numpy'])
    Ap_func = sympy.lambdify([parametersf, vars],Ap, modules=['numpy'])

    return Ja_func, Jp_func, Aa_func, Ap_func

def Lagrangian_as_function_of_jointcoordinates(robot_params):

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
    
    omega1 = sympy.diff(sympy.Matrix([0,0,theta1]),t)
    omega2 = sympy.diff(sympy.Matrix([0,0,theta2]),t)
    omega3 = sympy.diff(sympy.Matrix([0,0,theta3]),t)
    
    omega12 = sympy.diff(sympy.Matrix([0,0,phi12]),t)
    
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
    
    var = sympy.Matrix([d1, d2, d3, theta1, theta2, theta3, phi12, phi23, phi31])
    var_dot = sympy.diff(var, t)
    
    parameters_var = sympy.Matrix([rax, ray, rbx, rby, rcx, rcy, a1, a2, a3, b1, b2, b3, c12, c23, c31])
    geometric_var = sympy.Matrix([la1, la2, la3, lb1, lb2, lb3, ma1, ma2, ma3, mb1, mb2, mb3, mg, Ia1, Ia2, Ia3, Ib1, Ib2, Ib3, Ig])
    parameters_var_dict = dict(zip(parameters_var,robot_params.values()))
    geometric_var_dict = dict(zip(geometric_var, inertial_and_geometrical_parameters(robot_params).values()))
    L_reduced = L.subs(parameters_var_dict).subs(geometric_var_dict)
    
    L_func = sympy.lambdify([var,var_dot],L_reduced, modules=['numpy'])

    return L_func
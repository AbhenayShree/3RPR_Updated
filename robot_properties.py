def robot_parameters():
    rax = 0
    ray = 0
    rbx = 1
    rby = 0
    rcx = 1
    rcy = 2
    
    
    a1 = 0.1
    b1 = 0.3
    a2 = 0.1
    b2 = 0.3
    a3 = 0.1
    b3 = 0.3
    
    c12 = 0.15
    c23 = 0.15
    c31 = 0.15

    dictionary = {
        'rax': rax,
        'ray': ray,
        'rbx': rbx,
        'rby': rby,
        'rcx': rcx,
        'rcy': rcy,
        
        
        'a1': a1,
        'b1': b1,
        'a2': a2,
        'b2': b2,
        'a3': a3,
        'b3': b3,
        
        'c12': c12,
        'c23': c23,
        'c31': c31
    }

    return dictionary

def inertial_and_geometrical_parameters(robot_params):

    la1v = 0.75*robot_params['a1']
    la2v = 0.75*robot_params['a2']
    la3v = 0.75*robot_params['a3']

    lb1v = 0.5*robot_params['b1']
    lb2v = 0.5*robot_params['b2']
    lb3v = 0.5*robot_params['b3']

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

    Igv = mgv*0.5*robot_params['c12']**2

    inertial_and_geometrical_params = {
        'la1': la1v,
        'la2': la2v,
        'la3': la3v,
        'lb1': lb1v,
        'lb2': lb2v,
        'lb3': lb3v,
        'ma1': ma1v,
        'ma2': ma2v,
        'ma3': ma3v,
        'mb1': mb1v,
        'mb2': mb2v,
        'mb3': mb3v,
        'mg': mgv,
        'Ia1': Ia1v,
        'Ia2': Ia2v,
        'Ia3': Ia3v,
        'Ib1': Ib1v,
        'Ib2': Ib2v,
        'Ib3': Ib3v,
        'Ig': Igv
    }
    return inertial_and_geometrical_params
"""
A module with utitity functions for Thwaites
"""

def is_continuous(ufl):
    elem = ufl.ufl_element()

    # need to fix vectorial case:
    assert elem.value_size() == 1

    family = elem.family()
    if family == 'Lagrange':
        return True
    elif family == 'Discontinuous Lagrange' or family == 'DQ':
        return False
    else:
        raise NotImplemented('Unknown finite element family')

def normal_is_continuous(ufl):
    elem = ufl.ufl_element()

    family = elem.family()
    if family == 'Lagrange':
        return True
    elif family == 'Discontinuous Lagrange' or family == 'DQ':
        return False
    else:
        raise NotImplemented('Unknown finite element family')

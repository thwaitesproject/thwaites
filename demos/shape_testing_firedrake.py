from firedrake import *
from ufl import tensors
a = Constant(1.0)

b = as_matrix([[1,0],
              [0,1]])

def shape_function(x):
    print(x)
    if x.__class__ == Constant:
        print("True")
    elif x.__class__ == tensors.ListTensor:
        print("True")
    else:
        print(x.__class__)
        print("False")

shape_function(a)

shape_function(b)
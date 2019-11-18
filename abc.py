#!/usr/bin/python

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

'''
## Reaction (`.k` file)
    a + b <-> ab
    c + b <-> cb

## Mapping

    a	0	-1*v_0(y[0], y[1], y[2])
    b	1	-1*v_0(y[0], y[1], y[2])-1*v_1(y[4], y[3], y[1])
    ab	2	+1*v_0(y[0], y[1], y[2])
    c	3	-1*v_1(y[4], y[3], y[1])
    cb	4	+1*v_1(y[4], y[3], y[1])
'''


def reaction(y0, t, k0, k0r, k1, k1r):
    """
    Wrapper for the reaction.
    It receives an `np.array` `y0` with the initial concentrations and an
    `np.array` `t` with timestamps (it should also include `0`).
    This function solves the corresponding ODE system and returns an `np.array`
    `Y` in which each column represents a chemical species and each line a
    timestamp.
    """
    def dydt(y, t):
        return np.array([
                         -1*v_0(y[0], y[1], y[2]),
                         -1*v_0(y[0], y[1], y[2])-1*v_1(y[4], y[3], y[1]),
                         +1*v_0(y[0], y[1], y[2]),
                         -1*v_1(y[4], y[3], y[1]),
                         +1*v_1(y[4], y[3], y[1]),
                         ])
    
    # a + b <-> ab
    def v_0(a, b, ab):
        return k0 * a**1 * b**1 - k0r * ab**1
    
    # c + b <-> cb
    def v_1(cb, c, b):
        return k1 * c**1 * b**1 - k1r * cb**1
    
    return odeint(dydt, y0, t)

# Reaction rates:
# a + b <-> ab
k0 = .1
k0r = .1

# c + b <-> cb
k1 = .1
k1r = .1

# Initial concentrations:
y0 = np.array([
               100.0,  # a
               250.0,  # b
               0.0,  # ab
               200.0,  # c
               0.0,  # cb
               ])

t = np.linspace(0, .4, 100)
Y = reaction(y0, t, k0, k0r, k1, k1r)

plt.plot(t, Y[:, 0], label="a")
plt.plot(t, Y[:, 1], label="b")
plt.plot(t, Y[:, 2], label="ab")
plt.plot(t, Y[:, 3], label="c")
plt.plot(t, Y[:, 4], label="cb")
plt.legend(loc="best")
#plt.show()
plt.savefig('abc_umesh.png')
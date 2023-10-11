import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import math
import random

# parameters
N = 1000            # population size
B = 0.3             # rate of infektion
l = 1/7             # rate of recovery
y0 = [N-5, 5, 0]    # initial value for [Susepitble, Infekted, Recovered]
t_span = (0, 120)


def sir(t, y):
    S, I, R = y
    S + I + R
    dS = -B * (I/N) * S
    dI = B * (I/N) * S - l * I
    dR = l * I
    return dS, dI, dR


def prop(Y, coeff):
    # propensities
    S, I, R = Y
    B, l = coeff
    return B*I*S/N, l*I


def stoch():
    # stochiometry matrix
    return np.array([[-1, 1, 0], [0, -1, 1]])


def SSA(prop, stoch, X0, tspan, coeff):
    tvec = np.zeros(1)
    tvec[0] = tspan[0]
    Xarr = np.zeros([1, len(X0)])
    Xarr[0, :] = X0
    t = tvec[0]
    X = X0
    sMat = stoch()
    while t < tspan[1]:
        re = prop(X, coeff)
        a0 = sum(re)
        if a0 > 1e-10:
            r1 = random.random()
            r2 = random.random()
            tau = -math.log(r1)/a0
            cre = np.cumsum(re)
            cre = cre/cre[-1]
            r = 0
            while cre[r] < r2:
                r += 1
            t += tau
            tvec = np.append(tvec, t)
            X = X+sMat[r, :]
            Xarr = np.vstack([Xarr, X])
        else:
            print('Simulation ended at t=', t)
            return tvec, Xarr
    return tvec, Xarr


# get data using gillespie and plot
tvec, Xarr = SSA(prop, stoch, y0, t_span, [B, l])

plt.plot(tvec, Xarr[:, 0])
plt.plot(tvec, Xarr[:, 1])
plt.plot(tvec, Xarr[:, 2])
plt.xlabel('Tid (s)')
plt.ylabel('Personer (-)')
plt.title('Gillespie')
plt.legend(['S', 'I', 'R'])
# plt.savefig("sirGille.png")
plt.show()


# get data using solve_ivp and plot
times = np.arange(t_span[0], t_span[1], 0.01)
solution = sp.integrate.solve_ivp(sir, t_span, y0, t_eval=times)

plt.plot(solution.t, solution.y[0])
plt.plot(solution.t, solution.y[1])
plt.plot(solution.t, solution.y[2])
plt.xlabel('Tid (s)')
plt.ylabel('Personer (-)')
plt.title('Solve_ivp')
plt.legend(['S', 'I', 'R'])
# plt.savefig("sirIvp.png")
plt.show()

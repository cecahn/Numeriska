import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import math
import random

# parameters
N0 = 1000           # population size
B = 0.3             # rate of infektion
a = 0.2             # 1/a is the inkubation time
l = 1/7             # rate of recovery
u = 0.01            # mortality rate
v = 1.0             # rate of vaccination
# initial value for [Suscepitble, Exposed, Infekted, Recovered, Dead, Vaccinated]
y0 = [N0-5, 0, 5, 0, 0, 0]
t_span = (0, 120)


def seird(t, y):
    S, E, I, R, D, V = y
    N = N0-D
    dS = -B * (I/N) * S - v
    dE = B*(I/N)*S - a*E
    dI = a * E - l * I - u*I
    dR = l * I
    dD = u*I
    dV = v
    return dS, dE, dI, dR, dD, dV


def prop(Y, coeff):
    # propensities
    S, E, I, R, D, V = Y
    B, l, a, u, v = coeff
    N = N0-D
    return B*I*S/N, a*E, l*I, u*I, v


def stoch():
    # stochiometry matrix
    return np.array([[-1, 1, 0, 0, 0, 0], [0, -1, 1, 0, 0, 0], [0, 0, -1, 1, 0, 0], [0, 0, -1, 0, 1, 0], [-1, 0, 0, 0, 0, 1]])


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
tvec, Xarr = SSA(prop, stoch, y0, t_span, [B, l, a, u, v])

plt.plot(tvec, Xarr[:, 0])
plt.plot(tvec, Xarr[:, 1])
plt.plot(tvec, Xarr[:, 2])
plt.plot(tvec, Xarr[:, 3])
plt.plot(tvec, Xarr[:, 4])
plt.plot(tvec, Xarr[:, 5])
plt.xlabel('Tid (s)')
plt.ylabel('Personer (-)')
plt.title('Gillespie')
plt.legend(['S', 'E', 'I', 'R', 'D', 'V'])
# plt.savefig("seirdvGille.png")
plt.show()


# get data using solve_ivp and plot
times = np.arange(t_span[0], t_span[1], 0.01)
solution = sp.integrate.solve_ivp(seird, t_span, y0, t_eval=times)

plt.plot(solution.t, solution.y[0])
plt.plot(solution.t, solution.y[1])
plt.plot(solution.t, solution.y[2])
plt.plot(solution.t, solution.y[3])
plt.plot(solution.t, solution.y[4])
plt.plot(solution.t, solution.y[5])
plt.xlabel('Tid (s)')
plt.ylabel('Personer (-)')
plt.title('Solve_ivp')
plt.legend(['S', 'E', 'I', 'R', 'D', 'V'])
# plt.savefig("sierdvIvp.png")
plt.show()

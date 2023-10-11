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
v1 = 0.005           # rate of vaccination dose 1 (among susceptible)
v2 = 0.01            # rate of vaccination dose 2 (among non-immune dose 1)
r1 = 0.4            # immunity rate from dose 1
r2 = 0.7            # immunity rate from dose 2
s1 = 1 - r1         # chance of being susceptible after vaccine 1
s2 = 1 - r2         # chance of being suscepitble after vaccine 2
# initial value for [Suscepitble, Exposed, Infekted, Recovered, Immune through vaccine, Dead, Vaccinated dose 1, Vaccinated dose 2]
y0 = [N0-5, 0, 5, 0, 0, 0, 0, 0]
t_span = (0, 120)


def seird(t, y):
    S, E, I, R, Im, D, V1, V2 = y
    N = N0-D
    dS = -B*(I/N) * S - v1*S
    dE = B*(I/N) * S - a*E + B*(I/N) * (s1 + s2)
    dI = a*E - l*I - u*I
    dR = l*I
    dIm = r1*V1 + r2*V2
    dD = u*I
    dV1 = v1*S - r1*V1 - v2 * V1 - B*(I/N) * s1 * V1
    dV2 = v2 * V1 - r2*V2 - B*(I/N) * s2 * V2
    return dS, dE, dI, dR, dIm, dD, dV1, dV2


def prop(Y, coeff):
    # propensities
    S, E, I, R, Im, D, V1, V2 = Y
    B, l, a, u, v1, v2, r1, r2, s1, s2 = coeff
    N = N0-D
    return B*I*S/N, a*E, l*I, u*I, v1*S, r1*V1, v2*V1, r2*V2, B*I/N*s1*V1, B*I/N*s2*V2


def stoch():
    # stochiometry matrix
    return np.array([[-1, 1, 0, 0, 0, 0, 0, 0], [0, -1, 1, 0, 0, 0, 0, 0],
                     [0, 0, -1, 1, 0, 0, 0, 0], [0, 0, -1, 0, 0, 1, 0, 0],
                     [-1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, -1, 0],
                     [0, 0, 0, 0, 0, 0, -1, 1], [0, 0, 0, 0, 1, 0, 0, -1],
                     [0, 1, 0, 0, 0, 0, -1, 0], [0, 1, 0, 0, 0, 0, 0, -1]])


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
tvec, Xarr = SSA(prop, stoch, y0, t_span, [B, l, a, u, v1, v2, r1, r2, s1, s2])

plt.plot(tvec, Xarr[:, 0], color='blue')
plt.plot(tvec, Xarr[:, 1], linestyle='--', color='limegreen')
plt.plot(tvec, Xarr[:, 2], color='green')
plt.plot(tvec, Xarr[:, 3], color='red')
plt.plot(tvec, Xarr[:, 4], color='orange')
plt.plot(tvec, Xarr[:, 5], color='purple')
plt.plot(tvec, Xarr[:, 6], linestyle='--', color='thistle')
plt.plot(tvec, Xarr[:, 7], linestyle='--', color='plum')
plt.xlabel('Tid (s)')
plt.ylabel('Personer (-)')
plt.title('Gillespie')
plt.legend(['S', 'E', 'I', 'R', 'Im', 'D', 'V1', 'V2'])
# plt.savefig("seirimdvvGille.png")
plt.show()


# get data using solve_ivp and plot
times = np.arange(t_span[0], t_span[1], 0.01)
solution = sp.integrate.solve_ivp(seird, t_span, y0, t_eval=times)

plt.plot(solution.t, solution.y[0], color='blue')
plt.plot(solution.t, solution.y[1], linestyle='--', color='limegreen')
plt.plot(solution.t, solution.y[2], color='green')
plt.plot(solution.t, solution.y[3], color='red')
plt.plot(solution.t, solution.y[4], color='orange')
plt.plot(solution.t, solution.y[5], color='purple')
plt.plot(solution.t, solution.y[6], linestyle='--', color='thistle')
plt.plot(solution.t, solution.y[7], linestyle='--', color='plum')
plt.xlabel('Tid (s)')
plt.ylabel('Personer (-)')
plt.title('Solve_ivp')
plt.legend(['S', 'E', 'I', 'R', 'Im', 'D', 'V1', 'V2'])
# plt.savefig("seirimdvvIvp.png")
plt.show()

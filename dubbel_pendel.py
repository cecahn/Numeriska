import numpy as np
import scipy as sp
import math
from matplotlib import pyplot as plt
from matplotlib import animation

m = 1
l = 1

t0 = 0
t1 = 10

t_span = (t0, t1)
# vinkel 1, vinkel 2, rörelsemängd 1, rörelsemängd 2
y0 = (np.pi/10, np.pi/10, 0, 0)
times = np.arange(t0, t1, 0.1)


def fun(t, y, m, l):
    g = 9.81
    v1, v2, p1, p2 = y

    dv1 = (6/m/l/l)*((2*p1)-3*math.cos(v1-v2)*p2) / \
        (16-9*math.cos(v1-v2)**2*(v1-v2))
    dv2 = (6/m/l/l)*(8*p2-3*math.cos(v1-v2)*p1)/(16-9*math.cos(v1-v2)**2)
    dp1 = -(1/2)*m*l**2*(dv1*dv2*math.sin(v1-v2)+3*(g/l)*math.sin(v1))
    dp2 = (-1/2)*m*l**2*(-(dv1*dv2*math.sin(v1-v2))+(g/l)*math.sin(v2))
    return dv1, dv2, dp1, dp2


solution = sp.integrate.solve_ivp(fun, t_span, y0, t_eval=times, args=(m, l))

plt.plot(solution.t, solution.y[0])
plt.plot(solution.t, solution.y[1])
plt.title('En Plot')
plt.xlabel('t (sekunder)')
plt.ylabel('Vinkel (radianer)')
plt.show()

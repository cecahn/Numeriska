import numpy as np
import scipy as sp
import math
from matplotlib import pyplot as plt
from matplotlib import animation

m = 1
l = 1

t0 = 0
t1 = 10
h = 0.1

t_span = (t0, t1)
# y0 = vinkel 1, vinkel 2, rörelsemängd 1, rörelsemängd 2
y0 = (np.pi/2, np.pi/2, 0, 0)
times = np.arange(t0, t1, h)


def fun(t, y, m, l):
    g = 9.81
    v1, v2, p1, p2 = y

    dv1 = (6/m/l/l)*((2*p1)-3*math.cos(v1-v2)*p2) / \
        (16-9*math.cos(v1-v2)**2*(v1-v2))
    dv2 = (6/m/l/l)*(8*p2-3*math.cos(v1-v2)*p1)/(16-9*math.cos(v1-v2)**2)
    dp1 = -(1/2)*m*l**2*(dv1*dv2*math.sin(v1-v2)+3*(g/l)*math.sin(v1))
    dp2 = (-1/2)*m*l**2*(-(dv1*dv2*math.sin(v1-v2))+(g/l)*math.sin(v2))
    return dv1, dv2, dp1, dp2


# y = [old_v1, old_v2, old_p1, old_p2]
def Euler(y, h, m, l):
    dv1, dv2, dp1, dp2 = fun(0, y, m, l)
    v1 = y[0] + h * dv1
    v2 = y[1] + h * dv2
    p1 = y[2] + h * dp1
    p2 = y[3] + h * dp2
    return [v1, v2, p1, p2]


def fv1(m, l, y):
    v1, v2, p1, p2 = y
    dv1 = (6/m/l/l)*((2*p1)-3*math.cos(v1-v2)*p2) / \
        (16-9*math.cos(v1-v2)**2*(v1-v2))
    return dv1


def fv2(m, l, y):
    v1, v2, p1, p2 = y
    dv2 = (6/m/l/l)*(8*p2-3*math.cos(v1-v2)*p1)/(16-9*math.cos(v1-v2)**2)
    return dv2


def fp1(m, l, y, dv1, dv2):
    g = 9.81
    v1, v2, p1, p2 = y
    dp1 = -(1/2)*m*l**2*(dv1*dv2*math.sin(v1-v2)+3*(g/l)*math.sin(v1))
    return dp1


def fp2(m, l, y, dv1, dv2):
    g = 9.81
    v1, v2, p1, p2 = y
    dp2 = (-1/2)*m*l**2*(-(dv1*dv2*math.sin(v1-v2))+(g/l)*math.sin(v2))
    return dp2


def K(m, l, y):
    Kv1 = fv1(m, l, y)
    Kv2 = fv2(m, l, y)
    Kp1 = fp1(m, l, y, Kv1, Kv2)
    Kp2 = fp2(m, l, y, Kv1, Kv2)
    return Kv1, Kv2, Kp1, Kp2


def RungeKutta(y, h, m, l):
    K1v1, K1v2, K1p1, K1p2 = K(m, l, y)

    yK2 = [y[0] + h/2*K1v1, y[1] + h/2*K1v2, y[2] + h/2*K1p1, y[3] + h/2*K1p2]
    K2v1, K2v2, K2p1, K2p2 = K(m, l, yK2)

    yK3 = [y[0] + h/2*K2v1, y[1] + h/2*K2v2, y[2] + h/2*K2p1, y[3] + h/2*K2p2]
    K3v1, K3v2, K3p1, K3p2 = K(m, l, yK3)

    yK4 = [y[0] + h*K3v1, y[1] + h*K3v2, y[2] + h*K3p1, y[3] + h*K3p2]
    K4v1, K4v2, K4p1, K4p2 = K(m, l, yK4)

    v1_new = y[0] + h/6*(K1v1+2*K2v1+2*K3v1+K4v1)
    v2_new = y[1] + h/6*(K1v2+2*K2v2+2*K3v2+K4v2)
    p1_new = y[2] + h/6*(K1p1+2*K2p1+2*K3p1+K4p1)
    p2_new = y[3] + h/6*(K1p2+2*K2p2+2*K3p2+K4p2)

    return [v1_new, v2_new, p1_new, p2_new]


solution = sp.integrate.solve_ivp(fun, t_span, y0, t_eval=times, args=(m, l))

# Data for ploting Euler
i = 0
y = y0
v1_arr = [y[0]]
v2_arr = [y[1]]
t_arr = [0]
while i <= 10:
    # y = Euler(y, h, m, l)
    y = RungeKutta(y, h, m, l)
    v1_arr.append(y[0])
    v2_arr.append(y[1])
    t_arr.append(i)
    i += h

# plt.plot(solution.t, solution.y[0])
# plt.plot(solution.t, solution.y[1])
# plt.plot(t_arr, v1_arr)
# plt.plot(t_arr, v2_arr)
# plt.title('En Plot')
# plt.xlabel('t (sekunder)')
# plt.ylabel('Vinkel (radianer)')


# Animation
fig = plt.figure()
ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
ax.grid()
line, = ax.plot([], [], lw=2)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    x = np.cumsum([0,
                   1 * np.sin(v1_arr[i]),
                   1 * np.sin(v2_arr[i])])
    print(x)
    y = np.cumsum([0,
                   -1 * np.cos(v1_arr[i]),
                   -1 * np.cos(v2_arr[i])])

    line.set_data(x, y)
    return line,


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=300, interval=len(t_arr), blit=True)

plt.show()

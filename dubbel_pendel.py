import numpy as np
import scipy as sp
import math
from matplotlib import pyplot as plt
from matplotlib import animation

# parameters
m = 1       # pendulum mass
l = 1       # pendulum length
t0 = 0      # start time
t1 = 10     # end time
h = 0.1     # step size
# y0 = angle 1, angle 2, momentum 1, momentum 2
y0 = [np.pi/10, np.pi/10, 0, 0]
t_span = (t0, t1)


def plot(t, ys, xlab, ylab, text):
    for y in ys:
        plt.plot(t, y)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(text)
    plt.show()


def fun(t, y, m, l):
    # the differential equations we want to solve
    g = 9.81
    v1, v2, p1, p2 = y

    dv1 = (6/m/l/l)*((2*p1)-3*math.cos(v1-v2)*p2) / \
        (16-9*math.cos(v1-v2)**2*(v1-v2))
    dv2 = (6/m/l/l)*(8*p2-3*math.cos(v1-v2)*p1)/(16-9*math.cos(v1-v2)**2)
    dp1 = -(1/2)*m*l**2*(dv1*dv2*math.sin(v1-v2)+3*(g/l)*math.sin(v1))
    dp2 = (-1/2)*m*l**2*(-(dv1*dv2*math.sin(v1-v2))+(g/l)*math.sin(v2))
    return dv1, dv2, dp1, dp2


def Euler(t_span, y0, h, m, l):
    t = t_span[0]
    y = y0
    result = [[element] for element in ([t] + y)]

    while t <= t_span[1]:
        # update values
        dy = fun(t, y, m, l)
        y = [elem_y + h * elem_dy for elem_y, elem_dy in zip(y, dy)]
        t += h

        # add values to array for plotting
        result = [result[i] + [element]
                  for element, i in zip([t] + y, range(len([t] + y)))]

    return result


def RungeKutta(t_span, y0, h, m, l):
    t = t_span[0]
    y = y0
    result = [[element] for element in ([t] + y)]

    while t <= t_span[1]:
        # Get K1, K2, K3 and K4
        K1 = fun(t, y, m, l)

        y2 = [elem_y + h/2 * elem_K1 for elem_y, elem_K1 in zip(y, K1)]
        K2 = fun(t + h/2, y2, m, l)

        y3 = [elem_y + h/2 * elem_K2 for elem_y, elem_K2 in zip(y, K2)]
        K3 = fun(t + h/2, y3, m, l)

        y4 = [elem_y + h * elem_K3 for elem_y, elem_K3 in zip(y, K3)]
        K4 = fun(t + h, y4, m, l)

        # update values
        y = [elem_y + h/6 * (elem_K1 + 2*elem_K2 + 2*elem_K3 + elem_K4)
             for elem_y, elem_K1, elem_K2, elem_K3, elem_K4 in zip(y, K1, K2, K3, K4)]
        t += h

        # add values to array for plotting
        result = [result[i] + [element]
                  for element, i in zip([t] + y, range(len([t] + y)))]

    return result


# Uppgift 1
times = np.arange(t0, t1, h)
solution = sp.integrate.solve_ivp(fun, t_span, y0, t_eval=times, args=(m, l))
plot(solution.t, [solution.y[0], solution.y[1]],
     'Tid (sekunder)', 'Vinkel (radianer)', 'Solve_ivp')


# Uppgift 2
# Euler
h = 0.001
t_e, v1_e, v2_e, p1_e, p2_e = Euler(t_span, y0, h, m, l)
plot(t_e, [v1_e, v2_e],
     'Tid (sekunder)', 'Vinkel (radianer)', 'Euler')

# Runge Kutta
h = 0.1
t_rk, v1_rk, v2_rk, p1_rk, p2_rk = RungeKutta(t_span, y0, h, m, l)
plot(t_rk, [v1_rk, v2_rk],
     'Tid (sekunder)', 'Vinkel (radianer)', 'Runge-Kutta')


# Uppgift 3
# Set plot attriutes
fig = plt.figure()
ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
ax.grid()
line, = ax.plot([], [], lw=2)


def init():
    # initialize the animation
    line.set_data([], [])
    return line,


def animate(i):
    # animation step
    x = np.cumsum([0,
                   1 * np.sin(v1_rk[i]),
                   1 * np.sin(v2_rk[i])])
    print(x)
    y = np.cumsum([0,
                   -1 * np.cos(v1_rk[i]),
                   -1 * np.cos(v2_rk[i])])

    line.set_data(x, y)
    return line,


# plot animation
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=300, interval=len(t_rk), blit=True)
plot([], [], '', '', 'Pendel med Runge-Kutta')

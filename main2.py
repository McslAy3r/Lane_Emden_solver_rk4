import numpy as np
import matplotlib.pyplot as plt


def laneemden(xi, Y, n):
    """
    θ' = z
    z' = -2/ξ·z - θ^n
    """
    theta, z = Y
    # xi is "distance" from center if it's smaller than 1e-15 almso so it is almost zero, and we need to avoid dividing by zero 
    if abs(xi) < 1e-15:
        dtheta_d = z
        dz_dxi = -theta**n
    else: # ensuring that density is non negative
        theta_v = max(theta, 0.0)
        dtheta_d = z
        dz_dxi = -(2.0/xi)*z - theta_v**n
    return np.array([dtheta_d, dz_dxi])


# RK4
def rk4(func, xi, Y, h, n):
    
    k1 = func(xi, Y, n)
    k2 = func(xi + 0.5*h, Y + 0.5*h*k1, n)
    k3 = func(xi + 0.5*h, Y + 0.5*h*k2, n)
    k4 = func(xi + h,     Y + h*k3,     n)
    return Y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def integ(n, h=0.01, eps=1e-6, max_steps=100000):
    
    # integrating θ from ε until θ becomes non-positive
    xi = eps
    theta = 1.0 - eps**2/6.0
    z = -eps/3.0
    Y = np.array([theta, z])
    xi_vals = [xi]
    theta_vals = [theta]

    for step in range(max_steps): #stepping outward till we reach the surface
        Y = rk4(laneemden, xi, Y, h, n)
        xi += h
        xi_vals.append(xi)
        theta_vals.append(max(Y[0], 0.0)) 
        if Y[0] <= 0: # stopping when density becomes zero
            break

    return np.array(xi_vals), np.array(theta_vals)


def plot_single(n, h=0.01, eps=1e-6, max_steps=100000): #plotting for a single n
    
    xi_vals, theta_vals = integ(n, h=h, eps=eps, max_steps=max_steps)
    plt.figure(figsize=(6, 4))
    plt.plot(xi_vals, theta_vals, label=f'n = {n} (numerical)')
    # plotting analytical solution if input is n=0,1 or 5
    if np.isclose(n, 0):  # n=0, incompressible fluid (constant density star)
        xi0 = np.linspace(0, np.sqrt(6), 200)
        plt.plot(xi0, 1 - xi0**2 / 6, 'k--', label='n = 0 (analytical)')
    elif np.isclose(n, 1):# n=1, analytical solution exists
        xi1 = np.linspace(h, np.pi, 200)
        plt.plot(xi1, np.sin(xi1)/xi1, 'r--', label='n = 1 (analytical)')
    elif np.isclose(n, 5):# n=5, Represents a gas with infinite central concentration
        xi5 = np.linspace(0, 15, 500)
        plt.plot(xi5, (1 + xi5**2 / 3)**(-0.5), 'b--', label='n = 5 (analytical)')

    plt.xlabel('Radius ξ')
    plt.ylabel('Density θ(ξ)')
    plt.title(f'Lane–Emden for n = {n}')
    plt.grid(True)
    plt.legend()
    plt.show()

# plotting from n=0 to 5
plt.figure(figsize=(8, 6))

for n in range(6): 
    xi_vals, theta_vals = integ(n)
    plt.plot(xi_vals, theta_vals, label=f'n = {n} (numerical)')

# plotting available analytical solution where needed
xi0 = np.linspace(0, np.sqrt(6), 200)
plt.plot(xi0, 1 - xi0**2/6, 'k--', label='n = 0 (analytical)')

xi1 = np.linspace(0.01, np.pi, 200)
plt.plot(xi1, np.sin(xi1)/xi1, 'r--', label='n = 1 (analytical)')

xi5 = np.linspace(0, 15, 500)
plt.plot(xi5, (1 + xi5**2 / 3)**(-0.5), 'b--', label='n = 5 (analytical)')

plt.xlabel('Radius ξ')
plt.ylabel('Density θ(ξ)')
plt.title('Polytropic Stellar Structure for n = 0…5')
plt.xlim(0, 10)   
plt.ylim(0, 1.1)
plt.grid(True)
plt.legend()
plt.show()


# plot_single(5.6)
#  we can use this for plotting for a specific n

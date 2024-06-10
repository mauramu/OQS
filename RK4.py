import numpy as np
import scipy.constants as constants
import matplotlib.pyplot as plt


def runge_kutta_step(f, y, dt):
    k1 = f(y)
    k2 = f(y + dt / 2 * k1)
    k3 = f(y + dt / 2 * k2)
    k4 = f(y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# def master_equation(rho, H_s, gamma, n, sigma_minus, sigma_plus):
#     return -1j * (H_s @ rho - rho @ H_s) + gamma * (n + 1) * (sigma_minus @ rho @ sigma_plus - 1 / 2 * (
#             sigma_plus @ sigma_minus @ rho + rho @ sigma_plus @ sigma_minus)) + gamma * n * (
#             sigma_plus @ rho @ sigma_minus - 1 / 2 * (
#             sigma_minus @ sigma_plus @ rho + rho @ sigma_minus @ sigma_plus))

def commute(a, b):
    return (a @ b - b @ a)


def master_equation(rho, Hs, c_ops, c_ops_dag):
    res = -1j * commute(Hs, rho)
    for i in range(len(c_ops)):
        res += (c_ops[i] @ rho @ c_ops_dag[i] - 1 / 2 * (rho @ c_ops_dag[i] @ c_ops[i] + c_ops_dag[i] @ c_ops[i] @ rho))
    return res


# def hamiltonian(omega_a, omega_c, g, n):
#     return constants.hbar * np.array(
#         [[(n + 1) * omega_c, g * np.sqrt(n + 1)], [g * np.sqrt(n + 1), n * omega_c + omega_a]])

def hamiltonian_JC(w_a, w_c, g, sp, sm, adeg, a):
    return w_a * sp @ sm + w_c * adeg @ a + g * (adeg @ sm + sp @ a)


def n_distribution(T, omega):
    return 1 / (np.exp(constants.hbar * omega / (constants.k * T)) - 1)


def expectation_value(rho, operator):
    return np.trace(rho @ operator)


# %% two level rabi oscillations
# sz = np.array([[1, 0], [0, -1]])
sm = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])
# sm = np.array([[0, 0], [1, 0]])
sp = np.conj(sm).T
a = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
# a = np.array([[0, 0], [1, 0]])
adag = np.conj(a).T

omega_c = 1.0 * 2 * np.pi  # cavity frequency
omega_a = 1.0 * 2 * np.pi  # atom frequency
g_weak = 0.05 * 2 * np.pi  # coupling strength
g_strong = 0.5 * 2 * np.pi  # coupling strength
kappa = 0.1  # cavity dissipation rate
gamma = 0.1  # atom dissipation rate
n_th_a = .0  # temperature in frequency units

# collaps operators
c_op_list = []
c_op_dag_list = []

# Photon annihilation
rate = kappa * (1 + n_th_a)
c_op_list.append(np.sqrt(rate) * a)
c_op_dag_list.append(np.conj(np.sqrt(rate) * a).T)

# Photon creation
rate = kappa * n_th_a
c_op_list.append(np.sqrt(rate) * adag)
c_op_dag_list.append(np.conj(np.sqrt(rate) * adag).T)

# Atom annihilation
rate = gamma
c_op_list.append(np.sqrt(rate) * sm)
c_op_dag_list.append(np.conj(np.sqrt(rate) * sm).T)

H = hamiltonian_JC(omega_a, omega_c, g_weak, sp, sm, adag, a)

timestep = 0.01
time = np.arange(0, 50, timestep)

# rho_0 = np.array([[1, 0], [0, 0]], dtype=complex)
psi_0 = np.array([[1], [0], [0], [0]])
rho_0 = psi_0 @ np.conj(psi_0).T
# rho_0 = np.array([[1/2, 0], [0, 1/2]], dtype=complex)
solution = np.zeros((len(time), *np.shape(rho_0)), dtype=complex)
solution[0] = rho_0

for i in range(1, len(time)):
    solution[i] = runge_kutta_step(lambda rho: master_equation(rho, H, c_op_list, c_op_dag_list), solution[i - 1],
                                   timestep)

fig, ax = plt.subplots(1, 1)
# ax.plot(time, np.real(solution[:, 0, 0]), label=r'$\rho_{11}$')
# ax.plot(time, np.real(solution[:, 1, 1]), label=r'$\rho_{22}$')
# ax.plot(time, np.real(solution[:, 0, 1]), label=r'$\rho_{12}$')
# ax.plot(time, np.real(solution[:, 1, 0]), label=r'$\rho_{21}$')
ax.plot(time, [expectation_value(rho, adag @ a) for rho in solution], label=r'$\langle a^\dagger a \rangle$')
ax.plot(time, [expectation_value(rho, sp @ sm) for rho in solution], label=r'$\langle \sigma_- \sigma_+ \rangle$')
ax.legend()
plt.show()

# %% three level atom
# Levels
a_state = np.array([[1], [0], [0]])
b_state = np.array([[0], [1], [0]])
c_state = np.array([[0], [0], [1]])

# lambda model
# sigma_ab = a_state @ b_state.conj().T
# sigma_cb = c_state @ b_state.conj().T

# linear model
sigma_cb = b_state @ c_state.conj().T
sigma_ab = a_state @ b_state.conj().T

times = np.arange(0.0, 15.0, timestep)
Omega_1 = 1.0
Omega_2 = 1.0  # Drive
gamma_cb = 0.1
gamma_ba = 1  # Dissipation

# Hamiltonian
H = Omega_1 * (sigma_ab + sigma_ab.conj().T) + Omega_2 * (sigma_cb + sigma_cb.conj().T)

# Collapse operators (dissipation)
c_ops1 = [gamma_cb * sigma_cb, gamma_ba * sigma_ab]
c_ops1_dag = [gamma_cb * sigma_cb.conj().T, gamma_ba * sigma_ab.conj().T]

# Initial state
psi0 = a_state
rho0 = psi0 @ psi0.conj().T  # density matrix
sol = np.zeros((len(times), *np.shape(rho0)), dtype=complex)
sol[0] = rho0

for i in range(1, len(times)):
    sol[i] = runge_kutta_step(lambda rho: master_equation(rho, H, c_ops1, c_ops1_dag), sol[i - 1],
                              timestep)

fig, ax = plt.subplots(1, 1)
ax.plot(times, [expectation_value(rho, a_state @ a_state.conj().T) for rho in sol],
        label=r'$\langle a^\dagger a \rangle$')
ax.plot(times, [expectation_value(rho, b_state @ b_state.conj().T) for rho in sol],
        label=r'$\langle b^\dagger b \rangle$')
ax.plot(times, [expectation_value(rho, c_state @ c_state.conj().T) for rho in sol],
        label=r'$\langle c^\dagger c \rangle$')
ax.legend()
plt.show()

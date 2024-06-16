import numpy as np
import scipy.constants as constants
import matplotlib.pyplot as plt


def runge_kutta_step(f, y, dt):
    k1 = f(y)
    k2 = f(y + dt / 2 * k1)
    k3 = f(y + dt / 2 * k2)
    k4 = f(y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def commute(a, b):
    return (a @ b - b @ a)


def master_equation(rho, Hs, c_ops, c_ops_dag):
    res = -1j * commute(Hs, rho)
    for i in range(len(c_ops)):
        res += (c_ops[i] @ rho @ c_ops_dag[i] - 1 / 2 * (rho @ c_ops_dag[i] @ c_ops[i] + c_ops_dag[i] @ c_ops[i] @ rho))
    return res


def hamiltonian_JC(w_a, w_c, g, sp, sm, adeg, a):
    return w_a * sp @ sm + w_c * adeg @ a + g * (adeg @ sm + sp @ a)


def expectation_value(rho, operator):
    return np.trace(rho @ operator)


# %% two level rabi oscillations
sm = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])
sp = np.conj(sm).T
a = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
adag = np.conj(a).T

omega_c = 1.0 * 2 * np.pi  # cavity frequency
omega_a = 1.0 * 2 * np.pi  # atom frequency
g = [0.05 * 2 * np.pi, 0.1*2*np.pi]  # coupling strength
# g_strong = 0.5 * 2 * np.pi  # coupling strength
gamma = 0.1  # dissipation rate
n_th_a = 0.0  # temperature in frequency units

timestep = 0.01
time = np.arange(0, 30, timestep)
psi_0 = np.array([[1], [0], [0], [0]])
rho_0 = psi_0 @ np.conj(psi_0).T
solutions = np.zeros((len(time), *np.shape(rho_0), 2), dtype=complex)

for g_weak in g:
    # collaps operators
    c_op_list = []
    c_op_dag_list = []

    # Photon annihilation
    rate = gamma * (1 + n_th_a)
    c_op_list.append(np.sqrt(rate) * a)
    c_op_dag_list.append(np.conj(np.sqrt(rate) * a).T)

    # Photon creation
    rate = gamma * n_th_a
    c_op_list.append(np.sqrt(rate) * adag)
    c_op_dag_list.append(np.conj(np.sqrt(rate) * adag).T)

    # Atom annihilation
    rate = gamma
    c_op_list.append(np.sqrt(rate) * sm)
    c_op_dag_list.append(np.conj(np.sqrt(rate) * sm).T)

    H = hamiltonian_JC(omega_a, omega_c, g_weak, sp, sm, adag, a)
    solution = np.zeros((len(time), *np.shape(rho_0)), dtype=complex)
    solution[0] = rho_0
    for i in range(1, len(time)):
        solution[i] = runge_kutta_step(lambda rho: master_equation(rho, H, c_op_list, c_op_dag_list), solution[i - 1],
                                       timestep)
    solutions[:, :, :, g.index(g_weak)] = solution


fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(time, [expectation_value(rho, adag @ a) for rho in solutions[:, :, :, 0]], label=r'$\langle a^\dagger a \rangle$')
ax[0].plot(time, [expectation_value(rho, sp @ sm) for rho in solutions[:, :, :, 0]], label=r'$\langle \sigma_- \sigma_+ \rangle$')
# ax[0].axvline(np.pi/g_weak, linestyle='--', color='black')
ax[0].legend(frameon=False)
ax[0].set_title(r'g = 0.1$\pi$')
ax[0].set_ylabel('occupation probability')
ax[0].set_xlabel('t')
ax[1].plot(time, [expectation_value(rho, adag @ a) for rho in solutions[:, :, :, 1]], label=r'$\langle a^\dagger a \rangle$')
ax[1].plot(time, [expectation_value(rho, sp @ sm) for rho in solutions[:, :, :, 1]], label=r'$\langle \sigma_- \sigma_+ \rangle$')
# ax[1].axvline(np.pi/np.sqrt(g_weak+0.3*np.pi), linestyle='--', color='black')
ax[1].legend(frameon=False)
ax[1].set_title(r'g = 0.2$\pi$')
ax[1].set_ylabel('occupation probability')
ax[1].set_xlabel('t')
fig.tight_layout()
plt.savefig('two_level_rabi_coupling.png', dpi=300)
plt.show()

# %% three level atom
# Levels
a_state = np.array([[1], [0], [0]])
b_state = np.array([[0], [1], [0]])
c_state = np.array([[0], [0], [1]])
times = np.arange(0.0, 15.0, timestep)
Omega_1 = 1.0
Omega_2 = 1.0  # Drive
gammas_cb = [1, 0.3]
gammas_ba = [1, 0.3]

# linear model
# sigma_bc = b_state @ c_state.conj().T
# sigma_ab = a_state @ b_state.conj().T
# H = Omega_1 * (sigma_ab + sigma_ab.conj().T) + Omega_2 * (sigma_bc + sigma_bc.conj().T)

# lambda model
sigma_ab = a_state @ b_state.conj().T
sigma_bc = c_state @ b_state.conj().T
H = Omega_1 * (sigma_ab + sigma_ab.conj().T)

# Initial state
psi0 = a_state
rho0 = psi0 @ psi0.conj().T  # density matrix
sol = np.zeros((len(times), *np.shape(rho0)), dtype=complex)
sol[0] = rho0
solutions = np.zeros((*np.shape(sol), 4), dtype=complex)

for gamma_cb in gammas_cb:
    for gamma_ba in gammas_ba:
        print(gamma_cb, gamma_ba)
        # Collapse operators (dissipation)
        c_ops1 = [np.sqrt(gamma_cb) * sigma_bc, np.sqrt(gamma_ba) * sigma_ab]
        c_ops1_dag = [np.sqrt(gamma_cb) * sigma_bc.conj().T, np.sqrt(gamma_ba) * sigma_ab.conj().T]

        for i in range(1, len(times)):
            sol[i] = runge_kutta_step(lambda rho: master_equation(rho, H, c_ops1, c_ops1_dag), sol[i - 1],
                                  timestep)
        solutions[:, :, :, 2 * gammas_cb.index(gamma_cb) + gammas_ba.index(gamma_ba)] = sol


fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax[0, 0].plot(times, [expectation_value(rho, a_state @ a_state.conj().T) for rho in solutions[:, :, :, 0]],
              label=r'$\langle a^\dagger a \rangle$')
ax[0, 0].plot(times, [expectation_value(rho, b_state @ b_state.conj().T) for rho in solutions[:, :, :, 0]],
                label=r'$\langle b^\dagger b \rangle$')
ax[0, 0].plot(times, [expectation_value(rho, c_state @ c_state.conj().T) for rho in solutions[:, :, :, 0]],
                label=r'$\langle c^\dagger c \rangle$')
ax[0, 0].set_title(r'$\gamma_{cb} = %s, \gamma_{ba} = %s$' % (gammas_cb[0], gammas_ba[0]))
ax[0, 0].set_ylabel('occupation probability')
ax[0, 0].set_xlabel('t')
ax[0, 0].legend(frameon=False)
ax[0, 1].plot(times, [expectation_value(rho, a_state @ a_state.conj().T) for rho in solutions[:, :, :, 1]],
              label=r'$\langle a^\dagger a \rangle$')
ax[0, 1].plot(times, [expectation_value(rho, b_state @ b_state.conj().T) for rho in solutions[:, :, :, 1]],
                label=r'$\langle b^\dagger b \rangle$')
ax[0, 1].plot(times, [expectation_value(rho, c_state @ c_state.conj().T) for rho in solutions[:, :, :, 1]],
                label=r'$\langle c^\dagger c \rangle$')
ax[0, 1].set_title(r'$\gamma_{cb} = %s, \gamma_{ba} = %s$' % (gammas_cb[0], gammas_ba[1]))
ax[0, 1].set_ylabel('occupation probability')
ax[0, 1].set_xlabel('t')
ax[0, 1].legend(frameon=False)
ax[1, 0].plot(times, [expectation_value(rho, a_state @ a_state.conj().T) for rho in solutions[:, :, :, 2]],
              label=r'$\langle a^\dagger a \rangle$')
ax[1, 0].plot(times, [expectation_value(rho, b_state @ b_state.conj().T) for rho in solutions[:, :, :, 2]],
                label=r'$\langle b^\dagger b \rangle$')
ax[1, 0].plot(times, [expectation_value(rho, c_state @ c_state.conj().T) for rho in solutions[:, :, :, 2]],
                label=r'$\langle c^\dagger c \rangle$')
ax[1, 0].set_title(r'$\gamma_{cb} = %s, \gamma_{ba} = %s$' % (gammas_cb[1], gammas_ba[0]))
ax[1, 0].set_ylabel('occupation probability')
ax[1, 0].set_xlabel('t')
ax[1, 0].legend(frameon=False)
ax[1, 1].plot(times, [expectation_value(rho, a_state @ a_state.conj().T) for rho in solutions[:, :, :, 3]],
              label=r'$\langle a^\dagger a \rangle$')
ax[1, 1].plot(times, [expectation_value(rho, b_state @ b_state.conj().T) for rho in solutions[:, :, :, 3]],
                label=r'$\langle b^\dagger b \rangle$')
ax[1, 1].plot(times, [expectation_value(rho, c_state @ c_state.conj().T) for rho in solutions[:, :, :, 3]],
                label=r'$\langle c^\dagger c \rangle$')
ax[1, 1].set_title(r'$\gamma_{cb} = %s, \gamma_{ba} = %s$' % (gammas_cb[1], gammas_ba[1]))
ax[1, 1].set_ylabel('occupation probability')
ax[1, 1].set_xlabel('t')
ax[1, 1].legend(frameon=False)
fig.tight_layout()
plt.savefig('three_level_lambda_gamma.png', dpi=300)
plt.show()

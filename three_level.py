import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Levels
a_state = qt.basis(3, 0)
b_state = qt.basis(3, 1)
c_state = qt.basis(3, 2)
N = 2  # set truncation Fock state

# Operators
sigma_ab = qt.tensor(qt.qeye(N), a_state * b_state.dag())
sigma_cb = qt.tensor(qt.qeye(N), c_state * b_state.dag())
sigma_ac = qt.tensor(qt.qeye(N), a_state * c_state.dag())
N_a = qt.tensor(qt.qeye(N), a_state * a_state.dag())
N_b = qt.tensor(qt.qeye(N), b_state * b_state.dag())
N_c = qt.tensor(qt.qeye(N), c_state * c_state.dag())

times = np.linspace(0.0, 10.0, 200)
Omega = 1  # Drive
gamma_1 = 0.1  # Dissipation
gamma_2 = 0.1

# Hamiltonian
H = N_a + N_b + N_c + Omega * (sigma_ab + sigma_ab.dag() + sigma_cb + sigma_cb.dag() + sigma_ac)

# Collapse operators (dissipation)
# c_ops = [gamma_1 * sigma_ab, gamma_2 * sigma_cb, gamma_1 * sigma_ac]
c_ops = []
#
# Photon annihilation
rate = gamma_1 * (1 + 0)
c_ops.append(np.sqrt(rate) * a_state)

# Photon creation
rate = gamma_1 * 0
c_ops.append(np.sqrt(rate) * a_state.dag())

# Atom annihilation
rate = gamma_1
c_ops.append(np.sqrt(rate) * b_state)


# Operators for which to evaluate expectation values
expt_ops = [N_a, N_b, N_c]

# Initial state
psi0 = qt.tensor(qt.basis(N, 0), a_state)
rho0 = psi0 * psi0.dag()  # density matrix

#%%
result = qt.mesolve(H, rho0, times, c_ops, expt_ops)
result_states = qt.mesolve(H, rho0, times, c_ops)

plt.plot(times, result.expect[0])
plt.plot(times, result.expect[1])
plt.plot(times, result.expect[2])
plt.legend(("State 1", "State 2", "State 3"))
plt.show()

# %%
qt.plot_energy_levels([H, H + sigma_ab, H + sigma_cb])
plt.show()

# %%
qt.plot_wigner(result_states.states[0])
plt.title("Initial state")
plt.show()

qt.plot_wigner(result_states.states[-1])
plt.title("Final state")
plt.show()

qt.hinton(result_states.states[0])
plt.title("Initial state")
plt.show()

qt.hinton(result_states.states[-1])
plt.title("Final state")
plt.show()

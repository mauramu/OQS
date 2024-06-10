import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from qutip import about, basis, destroy, mesolve, qeye, sigmaz, tensor, plot_energy_levels

N = 2  # number of cavity fock states
wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.1 * 2 * np.pi  # atom frequency
g_weak = 0.05 * 2 * np.pi  # coupling strength
g_strong = 0.7 * 2 * np.pi  # coupling strength
gamma = 0.0  # atom dissipation rate
n_th_a = 0.0  # temperature in frequency units
use_rwa = True

tlist = np.linspace(0, 33, 1000)

psi0 = tensor(basis(N, 0), basis(2, 0))

# collapse operators
a = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), destroy(2).dag())
sz = tensor(qeye(N), sigmaz())


def h_atom():
    return wa * sm.dag() * sm


def h_cavity():
    return wc * a.dag() * a


def h_int(g):
    return g * (a.dag() * sm + a * sm.dag())


def hamiltonian(g):
    if use_rwa:
        # return wc * a.dag() * a + wa / 2 * sz + g * (a.dag() * sm + a * sm.dag())
        return wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + sm.dag() * a)


c_op_list = []

# Photon annihilation
rate = gamma * (1 + n_th_a)
c_op_list.append(np.sqrt(rate) * a)

# Photon creation
rate = gamma * n_th_a
c_op_list.append(np.sqrt(rate) * a.dag())

# Atom annihilation
rate = gamma
c_op_list.append(np.sqrt(rate) * sm)

output = mesolve(hamiltonian(g_weak), psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm])
# output_states = mesolve(hamiltonian(g_weak), psi0, tlist, c_op_list)

# %%
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(tlist, output.expect[0], label="Cavity excitation")
ax[0].plot(tlist, output.expect[1], label="Atom excitation")
ax[0].set_xlabel("t")
ax[0].set_ylabel("<n>")
# ax[1].plot(tlist, output1.expect[0], label="Cavity excitation")
# ax[1].plot(tlist, output1.expect[1], label="Atom excitation")
ax[1].legend(frameon=False)
ax[1].set_xlabel("t")
# ax[1].set_ylabel("<n>")
plt.tight_layout()
plt.savefig('temperature.png', dpi=300)
plt.show()

# %%
plot_energy_levels([hamiltonian(0), hamiltonian(g_weak), hamiltonian(g_strong)],
                   h_labels=["no coupling", "weak", "strong"])
plt.title("Energy levels of the coupled cavity-atom system")
plt.savefig('energy_levels.png', dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
plot_energy_levels([h_atom(), h_cavity(), h_int(g_weak)], [r'$H_a$', r'$H_c$', r'$H_{int}$'],
                   ax=ax[0])
plot_energy_levels([hamiltonian(0), hamiltonian(g_weak), hamiltonian(g_strong)],
                   h_labels=["$g=0$", f"$g={np.round(g_weak, 2)}$", f"$g={np.round(g_strong, 2)}$"], ax=ax[1])
plt.tight_layout()
plt.savefig('energy_levels_interaction.png', dpi=300)
plt.show()
# %%
qt.hinton(output_states.states[-1].ptrace(0))
plt.title("Final cavity state")
plt.show()

qt.hinton(output_states.states[-1].ptrace(1))
plt.title("Final atom state")
plt.show()

# %%
qt.plot_wigner(output_states.states[-1].ptrace(0))
plt.title("Wigner function of final cavity state")
plt.show()

qt.plot_wigner(output_states.states[-1].ptrace(1))
plt.title("Wigner function of final atom state")
plt.show()

qt.plot_wigner(output_states.states[-1])
plt.title("Wigner function of final state")
plt.show()

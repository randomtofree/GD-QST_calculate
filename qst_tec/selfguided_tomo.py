import numpy as np
from numpy.random import default_rng

import qutip as qtp
from qutip import basis, tensor
from qutip import coherent, coherent_dm, expect, Qobj, fidelity, rand_dm
from qutip.wigner import wigner, qfunc



import optax

from tqdm.auto import tqdm
import time

from examples.nqudits_tomo import gellmann

def gellmann_matrices(d):
    matrices = []

    # Off-diagonal symmetric matrices: E_ij + E_ji, for 0 <= i < j < d
    for i in range(d):
        for j in range(i+1, d):
            mat = np.zeros((d, d), dtype=complex)
            mat[i, j] = 1
            mat[j, i] = 1
            matrices.append(mat)

    # Off-diagonal antisymmetric matrices: -i(E_ij - E_ji)
    for i in range(d):
        for j in range(i+1, d):
            mat = np.zeros((d, d), dtype=complex)
            mat[i, j] = -1j
            mat[j, i] = 1j
            matrices.append(mat)

    # Diagonal matrices
    for k in range(1, d):
        mat = np.zeros((d, d), dtype=complex)
        coeff = np.sqrt(2 / (k * (k + 1)))
        for i in range(k):
            mat[i, i] = 1
        mat[k, k] = -k
        matrices.append(coeff * mat)

    return matrices


def selfguided_tomo(data, rho_or, iterations: int,  batch_size: int,
            a=3, b = 0.1, A =0, s=0.602, t=0.101, batch=True, tqdm_off=False):
  """
  Function to do the GD-Chol.
  Return:
    params1: The reconstructed density matrix
    fidelities_GD: A list with the fidelities values per iteration
    timel_GD: A list with the value of the time per iteration
    loss1: A list with the value of the loss function per iteration

  Input:
    data: the expected value of the original density matrix
    rho_or: original density matrix, to calculate the fidelity
    ops_jnp: POVM in jnp array
    params: Ansatz, any complex matrix T (not necessary the lower triangular)
    iterations: number of iterations for the method
    batch_size: batch size
    lr: learning rate
    decay: value of the decay of the lr (defined in given optimizer)
    lamb: hyperparameter l1 regularization
    batch: True to have mini batches, False to take all the data
    tqdm_off: To show the iteration bar. True is to desactivate (for the cluster)
    
  """
  local_unitary = Matrix

  loss1 = []
  fidelities_GD = []
  timel_GD = []
  #par_o = jnp.matmul(jnp.conj(params.T),params)/jnp.trace(jnp.matmul(jnp.conj(params.T),params))
  #fidelities_GD.append(qtp.fidelity(rho_or, qtp.Qobj(par_o)))
  #loss1.append(float(cost(params, jnp.asarray(data), ops_jnp, lamb)))
  opt_state = gradient_transform.init(params)
  num_me = len(data)
  # opt_state = optimizer.init(params)
  if not tqdm_off:
    pbar_GD = tqdm(range(iterations)) 
  
  @jit
  def step(params, opt_state, data, ops_jnp):
    grad_f = jax.grad(cost, argnums=0)(params, data, ops_jnp, lamb)
    grads = jnp.conj(grad_f)           # do a conjugate, if not can create some problems
    # updates, opt_state = optimizer.update(grads, opt_state, params)
    updates, opt_state = gradient_transform.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state
  

  tot_time = 0
  for i in tqdm(range(iterations), disable=tqdm_off):
    start = time.time()
    if batch:
        rng = default_rng()
        indix = rng.choice(num_me, size=batch_size, replace=False)
        # indix = np.random.randint(0, num_me, size=[batch_size])
        data_b = jnp.asarray(data[[indix]].flatten())
        ops2 = ops_jnp[indix]
    else: 
        ops2 = ops_jnp
        data_b = data
    params, opt_state = step(params, opt_state,data_b, ops2)
    #params = rho_cons(params)
    par1 = jnp.matmul(jnp.conj(params.T),params)/jnp.trace(jnp.matmul(jnp.conj(params.T),params))
    loss1.append(float(cost(params, data_b, ops2, lamb)))
    f = qtp.fidelity(rho_or, qtp.Qobj(par1))
    fidelities_GD.append(f)
    
    end = time.time()
    timestep = end - start
    tot_time += timestep
    timel_GD.append(tot_time)
    #timel_GD.append(end - start)  
    if not tqdm_off:
        pbar_GD.set_description("Fidelity GD-chol-rank {:.4f}".format(f))
        pbar_GD.update()

  params1 = jnp.matmul(jnp.conj(params.T),params)/jnp.trace(jnp.matmul(jnp.conj(params.T),params))
  return params1, fidelities_GD, timel_GD, loss1
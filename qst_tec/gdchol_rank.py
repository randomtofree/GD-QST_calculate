import sys

import numpy as np
from numpy.random import default_rng

import qutip as qtp
from qutip import basis, tensor
from qutip import coherent, coherent_dm, expect, Qobj, fidelity, rand_dm
from qutip.wigner import wigner, qfunc

import jax
import jax.numpy as jnp
import jax.numpy.linalg  as nlg
from jax import grad
from jax import jit
from jax.example_libraries import optimizers
from jax import config
config.update("jax_enable_x64", True)

import optax

from tqdm.auto import tqdm
import time




@jit
def calculate_ccnr(rho: jnp.ndarray):
    """
    计算双体系统密度矩阵的 CCNR (Realignment) 值
    rho: shape (D, D) 的密度矩阵，其中 D = d_local**2
    """
    # 1. 自动获取维度
    D = rho.shape[0]
    # 使用 int() 确保 d 是静态整数，避免 JIT 报错
    d = int(D**0.5) 
    
    # 2. Reshape 为 4 阶张量 (m, mu, n, nu)
    tensor_rho = rho.reshape((d, d, d, d))
    
    # 3. Realignment 操作: (m, mu, n, nu) -> (m, n, mu, nu)
    realigned_tensor = tensor_rho.transpose((0, 2, 1, 3))
    
    # 4. 压平回矩阵进行 SVD
    realigned_matrix = realigned_tensor.reshape((D, D))
    
    # 5. 计算核范数 (Singular values 的和)
    singular_values = jnp.linalg.svd(realigned_matrix, compute_uv=False)
    
    return jnp.sum(singular_values)


@jit
def partial_transpose_b_penalty(rho: jnp.ndarray):
    """
    对 d*d 系统的第二个子系统 (Bob) 进行偏转置并返回 NPT 惩罚值
    """
    # 1. 自动获取维度
    D = rho.shape[0]
    # 使用 int() 确保 d 是静态整数，避免 JIT 报错
    d = int(D**0.5) 
    
    # 2. 重塑并偏转置
    # (iA, iB, jA, jB) -> (iA, jB, jA, iB)
    rho_pt = rho.reshape((d, d, d, d)).transpose((0, 3, 2, 1)).reshape((D, D))
    
    # 3. 计算特征值 (由于 rho 是 Hermite 的，PT 后依然是 Hermite 的)
    # eigh 只计算特征值，[0] 拿到 evals
    evals = jnp.linalg.eigh(rho_pt)[0]
    
    # 4. 惩罚逻辑
    # 如果 evals >= 0 (PPT), 则返回 0
    # 如果 evals < 0 (NPT), 则返回负值的平方和
    # 建议加入一个极小的 epsilon 提高数值稳定性
    neg_evals = jnp.maximum(0.0, -evals)
    return jnp.sum(jnp.square(neg_evals))


@jit
def cost(rho1: jnp.ndarray,  lamb:float):
    """
    Return the cost function to do GD 
    rho1: initial guess T (any complex matrix)
    ops_jnp: measurement operator array in jax format
    data: list of expectation values (real numbers), usually an experimental data set
    """
    # print(len(jnpexpect(Oper,rho1)))
    t_dag_t = jnp.conj(rho1.T) @ rho1
    rho = t_dag_t / jnp.trace(t_dag_t)
    l2 = 3/(4*calculate_ccnr(rho)-1)
    return l2 + lamb*partial_transpose_b_penalty(rho1)



def gd_chol_rank(params: optax.Params, iterations: int,  batch_size: int,
            lr=2e-1, decay = 0.999, lamb:float =0.00001, batch=True, tqdm_off=False):
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
  start_learning_rate = lr
  # Exponential decay of the learning rate.
  scheduler = optax.exponential_decay(
      init_value=start_learning_rate, 
      transition_steps=iterations,
      decay_rate=decay)
  # Combining gradient transforms using `optax.chain`.
  gradient_transform = optax.chain(
      optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
      optax.scale_by_adam(),  # Use the updates from adam.
      optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
      # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
      optax.scale(-1.0)
  )
  

  loss1 = []
  purity_GD = []
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
  def step(params, opt_state):
    grad_f = jax.grad(cost, argnums=0)(params, lamb)
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
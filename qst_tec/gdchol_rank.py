import sys

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit
from jax import config
config.update("jax_enable_x64", True)

import optax

from tqdm.auto import tqdm
import time
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================================================================
# Custom forward/backward passes (bypass JAX's expensive default SVD/eigh autodiff)
# ==============================================================================

# 1. Efficient gradient for CCNR nuclear norm
@jax.custom_vjp
def custom_nuclear_norm(M):
    s = jnp.linalg.svd(M, compute_uv=False)
    return jnp.sum(s)

def nuclear_norm_fwd(M):
    U, s, Vh = jnp.linalg.svd(M, full_matrices=False)
    return jnp.sum(s), (U, Vh)

def nuclear_norm_bwd(res, g):
    U, Vh = res
    # Return the complex-conjugate of the analytic gradient (JAX convention)
    grad_M = g * (U @ Vh).conj()
    return (grad_M,)

custom_nuclear_norm.defvjp(nuclear_norm_fwd, nuclear_norm_bwd)


# 2. Efficient gradient for PPT penalty
@jax.custom_vjp
def custom_ppt_penalty(rho_pt):
    evals = jnp.linalg.eigvalsh(rho_pt)
    return jnp.sum(jnp.minimum(0., evals)**2)

def ppt_penalty_fwd(rho_pt):
    evals, evecs = jnp.linalg.eigh(rho_pt)
    penalty = jnp.sum(jnp.minimum(0., evals)**2)
    return penalty, (evals, evecs)

def ppt_penalty_bwd(res, g):
    evals, evecs = res
    grad_evals = jnp.where(evals < 0, 2.0 * evals, 0.0)
    G = evecs @ jnp.diag(grad_evals) @ evecs.conj().T
    # Return the complex-conjugate of the analytic gradient
    grad_matrix = g * G.conj()
    return (grad_matrix,)

custom_ppt_penalty.defvjp(ppt_penalty_fwd, ppt_penalty_bwd)

@jit
def calculate_ccnr(rho):
    """Compute the Computable Cross-Norm Robustness (CCNR) of density matrix rho."""
    d = int(rho.shape[0] ** 0.5)
    rho_tensor = rho.reshape((d, d, d, d))
    # Realign tensor
    rho_R_tensor = jnp.transpose(rho_tensor, (0, 2, 1, 3))
    rho_R = rho_R_tensor.reshape((d**2, d**2))
    return custom_nuclear_norm(rho_R)


@jit
def partial_transpose_b_penalty(rho):
    """Compute the partial transpose (PPT) penalty for subsystem B."""
    d = int(rho.shape[0] ** 0.5)
    rho_tensor = rho.reshape((d, d, d, d))
    # Partial transpose: swap row_B (axis 1) and col_B (axis 3)
    rho_pt_tensor = jnp.transpose(rho_tensor, (0, 3, 2, 1))
    rho_pt = rho_pt_tensor.reshape((d**2, d**2))

    # Enforce Hermitian symmetry (prevents eigh errors from floating-point drift)
    rho_pt = (rho_pt + rho_pt.conj().T) / 2.0
    return custom_ppt_penalty(rho_pt)


@jit
def cost(rho1: jnp.ndarray, lamb: float):
    """
    Cost function for GD-based BES search.

    Args:
        rho1: initial guess T (arbitrary complex matrix, shape [rank, D])
        lamb: penalty weight for PPT violation

    Returns:
        Negative CCNR + lamb * PPT penalty (minimizing this pushes CCNR > 1 while keeping PPT ≥ 0).
    """
    t_dag_t = jnp.conj(rho1.T) @ rho1
    rho = t_dag_t / jnp.trace(t_dag_t)
    l2 = -calculate_ccnr(rho)
    return l2 + lamb * partial_transpose_b_penalty(rho)


def gd_chol_bes_search(params: optax.Params, iterations: int,
                       lr=2e-1, decay=0.01,
                       lamb: float=0.01, lamb_end: float=500,
                       tqdm_off=False, record_freq=100):
    """
    GD-based Bound Entangled State (BES) search.

    Uses jax.lax.scan with chunking for memory-efficient iteration over
    dynamically annealed penalty coefficient (lambda).

    Args:
        params: initial ansatz T (complex JAX array, shape [rank, D]).
        iterations: total number of GD steps.
        lr: initial learning rate.
        decay: exponential decay rate for LR scheduler.
        lamb: initial PPT penalty weight.
        lamb_end: final PPT penalty weight (linear annealing). If None, lamb is constant.
        tqdm_off: suppress tqdm progress bar.
        record_freq: metric recording frequency (chunk size).

    Returns:
        (final_rho, ccnr_track, ppt_track, timel_GD, loss_track)
    """
    if lamb_end is None:
        lamb_end = lamb

    # Pre-compute lambda array (linear annealing); swap to jnp.logspace for exponential
    lamb_array = jnp.linspace(lamb, lamb_end, iterations)

    start_learning_rate = lr
    scheduler = optax.exponential_decay(
        init_value=start_learning_rate, 
        transition_steps=iterations,
        decay_rate=decay)
        
    gradient_transform = optax.chain(
        optax.clip_by_global_norm(1.0),  
        optax.scale_by_adam(),          
        optax.scale_by_schedule(scheduler), 
        optax.scale(-1.0) 
    )
  
    loss1 = []
    ccnr_track = [] 
    ppt_track = []   
    timel_GD = []
    
    opt_state = gradient_transform.init(params)
    
    @jax.jit
    def compute_metrics(current_params, current_lamb):
        t_dag_t = jnp.matmul(jnp.conj(current_params.T), current_params)
        rho = t_dag_t / jnp.trace(t_dag_t)
        
        loss_val = cost(current_params, current_lamb)
        ccnr_val = calculate_ccnr(rho)
        ppt_val = partial_transpose_b_penalty(rho)
        return loss_val, ccnr_val, ppt_val

    def scan_step(state, current_lamb):
        current_params, current_opt_state = state
        loss_val, grad_f = jax.value_and_grad(cost, argnums=0)(current_params, current_lamb)
        grads = jnp.conj(grad_f)
        updates, new_opt_state = gradient_transform.update(grads, current_opt_state, current_params)
        new_params = optax.apply_updates(current_params, updates)
        return (new_params, new_opt_state), None

    chunk_size = record_freq
    num_chunks = iterations // chunk_size
    remainder = iterations % chunk_size

    @jax.jit
    def scan_chunk(current_state, lamb_chunk):
        return jax.lax.scan(scan_step, current_state, lamb_chunk)

    tot_time = 0
    current_state = (params, opt_state)
    total_steps = num_chunks + (1 if remainder > 0 else 0)

    pbar_GD = None if tqdm_off else tqdm(total=iterations, desc="BES Search")

    for i in range(total_steps):
        start = time.time()

        if i < num_chunks:
            chunk_lambdas = lamb_array[i * chunk_size : (i + 1) * chunk_size]
            current_state, _ = scan_chunk(current_state, chunk_lambdas)
            steps_run = chunk_size
        else:
            chunk_lambdas = lamb_array[num_chunks * chunk_size : ]
            current_state, _ = scan_chunk(current_state, chunk_lambdas)
            steps_run = remainder

        current_params = current_state[0]

        # Use the last lambda value in this chunk for metric reporting
        current_lamb_val = float(chunk_lambdas[-1])
        loss_val, ccnr_val, ppt_val = compute_metrics(current_params, current_lamb_val)

        current_loss = float(loss_val)
        current_ccnr = float(ccnr_val)
        current_ppt = float(ppt_val)

        end = time.time()
        tot_time += (end - start)

        loss1.append(current_loss)
        ccnr_track.append(current_ccnr)
        ppt_track.append(current_ppt)
        timel_GD.append(tot_time)

        if pbar_GD is not None:
            pbar_GD.update(steps_run)
            pbar_GD.set_postfix({
                "Loss": f"{current_loss:.4f}",
                "CCNR": f"{current_ccnr:.4f}",
                "PPT": f"{current_ppt:.2e}",
                "Lamb": f"{current_lamb_val:.2f}"
            })

    if pbar_GD is not None:
        pbar_GD.close()

    # Reconstruct final density matrix
    final_params = current_state[0]
    params1 = jnp.matmul(jnp.conj(final_params.T), final_params) / jnp.trace(jnp.matmul(jnp.conj(final_params.T), final_params))
    
    return params1, ccnr_track, ppt_track, timel_GD, loss1

def plot_density_matrix_heatmap(rho, title="Searched Density Matrix"):
    """Plot real and imaginary parts of a density matrix as heatmaps."""
    rho_np = np.array(rho)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    vmax = max(np.max(np.abs(np.real(rho_np))), np.max(np.abs(np.imag(rho_np))))
    vmin = -vmax

    sns.heatmap(np.real(rho_np), ax=axes[0], cmap="RdBu_r",
                center=0, vmin=vmin, vmax=vmax,
                annot=False, cbar=True, square=True)
    axes[0].set_title(f"{title} - Real Part")

    sns.heatmap(np.imag(rho_np), ax=axes[1], cmap="RdBu_r",
                center=0, vmin=vmin, vmax=vmax,
                annot=False, cbar=True, square=True)
    axes[1].set_title(f"{title} - Imaginary Part")

    plt.tight_layout()
    plt.show()
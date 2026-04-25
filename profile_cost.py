import sys
sys.path.insert(0, '/Users/zwh/vscodeprojects/GD-QST_calculate')

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import time

from qst_tec.gdchol_rank import calculate_ccnr, partial_transpose_b_penalty, cost

# ---- Parameters ----
rank = 7
D = 16           # 4x4 bipartite system: d_local=4, D = d_local^2 = 16
lamb = 0.01
n_runs = 100

# ---- Create random complex ansatz T (rank, D) ----
key = jax.random.PRNGKey(42)
key_real, key_imag = jax.random.split(key)
T_real = jax.random.normal(key_real, (rank, D), dtype=jnp.float64)
T_imag = jax.random.normal(key_imag, (rank, D), dtype=jnp.float64)
T = T_real + 1j * T_imag

# Pre-compute rho = T^dag T / tr(T^dag T) for sub-function benchmarks
t_dag_t = jnp.conj(T.T) @ T
rho = t_dag_t / jnp.trace(t_dag_t)

print(f"T shape:       {T.shape}, dtype: {T.dtype}")
print(f"rho shape:     {rho.shape}, dtype: {rho.dtype}")
print()

# ---- Warmup (includes JIT compilation) ----
print("Warming up (includes JIT compilation)...", flush=True)
t0 = time.perf_counter()
_ = calculate_ccnr(rho).block_until_ready()
_ = partial_transpose_b_penalty(rho).block_until_ready()
_ = cost(T, lamb).block_until_ready()
_ = jax.value_and_grad(cost)(T, lamb)[0].block_until_ready()
_ = jax.value_and_grad(cost)(T, lamb)[1].block_until_ready()
print(f"Warmup done in {time.perf_counter() - t0:.2f} s.\n")

def bench(name, fn, *args):
    start = time.perf_counter()
    for _ in range(n_runs):
        out = fn(*args)
        if isinstance(out, tuple):
            for o in out:
                jax.block_until_ready(o)
        else:
            jax.block_until_ready(out)
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / n_runs * 1000.0
    print(f"  {name:<40s} {avg_ms:8.3f} ms per call")
    return avg_ms

print("=" * 64)
print("BENCHMARKS ({} runs each, after warmup)".format(n_runs))
print("=" * 64)

t_ccnr = bench("calculate_ccnr(rho)",            calculate_ccnr, rho)
t_ppt  = bench("partial_transpose_b_penalty(rho)", partial_transpose_b_penalty, rho)
t_cost = bench("cost(T, lamb)",                  cost, T, lamb)
t_grad = bench("jax.value_and_grad(cost)(T, lamb)", lambda t, l: jax.value_and_grad(cost)(t, l), T, lamb)

# ---- Summary ----
print()
print("=" * 64)
print("COST BREAKDOWN SUMMARY")
print("=" * 64)
print(f"  calculate_ccnr:               {t_ccnr:8.3f} ms  ({t_ccnr / t_cost * 100:5.1f}% of cost)")
print(f"  partial_transpose_b_penalty:  {t_ppt:8.3f} ms  ({t_ppt / t_cost * 100:5.1f}% of cost)")
print(f"  cost (total):                 {t_cost:8.3f} ms  (100%)")
print()
print(f"  ccnr + ppt (standalone):      {t_ccnr + t_ppt:8.3f} ms  (ratio vs cost: {(t_ccnr + t_ppt) / t_cost:.2f}x)")

print()
print("=" * 64)
print("GRADIENT OVERHEAD")
print("=" * 64)
print(f"  value_and_grad(cost):         {t_grad:8.3f} ms")
print(f"  Gradient overhead:            {t_grad - t_cost:8.3f} ms  ({t_grad / t_cost:.2f}x the forward pass)")
print()

print("=" * 64)
print("PER-ITERATION EXTRAPOLATION (gd_chol_bes_search)")
print("=" * 64)
print(f"  One grad step (value_and_grad + optax update): ~{t_grad:.3f} ms")
print(f"    1 000 iterations:   ~{t_grad * 1_000 / 1000:.1f} s")
print(f"    5 000 iterations:   ~{t_grad * 5_000 / 1000:.1f} s")
print(f"   10 000 iterations:   ~{t_grad * 10_000 / 1000:.1f} s")
print(f"  100 000 iterations:   ~{t_grad * 100_000 / 1000:.1f} s")

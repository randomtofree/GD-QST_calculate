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
import seaborn as sns
import matplotlib.pyplot as plt



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
    l2 = -calculate_ccnr(rho)
    return l2 + lamb*partial_transpose_b_penalty(rho)


def gd_chol_bes_search(params: optax.Params, iterations: int,
                       lr=2e-1, decay=0.01, 
                       lamb: float=0.01, lamb_end: float=500, # 新增 lamb_end 保持兼容
                       tqdm_off=False, record_freq=100):
    """
    修改自 GD-QST，专用于搜索 Bound Entangled States (BES)。
    采用 jax.lax.scan 与 chunking 机制重构，支持动态惩罚系数退火。
    输出接口与原版完全一致。
    """
    # 【兼容性处理】：如果未提供 lamb_end，则退化为恒定 lamb
    if lamb_end is None:
        lamb_end = lamb
    
    # 预先生成整个迭代过程的 lambda 数组 (线性增长)
    # 你也可以根据需要改成 jnp.logspace 实现指数增长
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

    # 【修改】：scan_step 现在的第二个输入变成了 current_lamb，而不是无用的 step_idx
    def scan_step(state, current_lamb):
        current_params, current_opt_state = state
        
        # 将 current_lamb 动态传入代价函数
        loss_val, grad_f = jax.value_and_grad(cost, argnums=0)(current_params, current_lamb)
        grads = jnp.conj(grad_f)
        
        updates, new_opt_state = gradient_transform.update(grads, current_opt_state, current_params)
        new_params = optax.apply_updates(current_params, updates)
        
        return (new_params, new_opt_state), None

    chunk_size = record_freq
    num_chunks = iterations // chunk_size
    remainder = iterations % chunk_size

    # 【修改】：scan_chunk 现在需要接收一段 lamb_chunk 数组作为遍历对象
    @jax.jit
    def scan_chunk(current_state, lamb_chunk):
        return jax.lax.scan(scan_step, current_state, lamb_chunk)
        
    @jax.jit
    def scan_remainder(current_state, lamb_chunk):
        return jax.lax.scan(scan_step, current_state, lamb_chunk)

    tot_time = 0
    current_state = (params, opt_state)
    total_steps = num_chunks + (1 if remainder > 0 else 0)
    
    # 【UI 优化】：使用 total=iterations 让进度条显示真实的总步数
    pbar_GD = None if tqdm_off else tqdm(total=iterations, desc="BES Search")

    for i in range(total_steps):
        start = time.time()
        
        # 1. 切片获取当前 Chunk 对应的 lambda 数组
        if i < num_chunks:
            chunk_lambdas = lamb_array[i * chunk_size : (i + 1) * chunk_size]
            current_state, _ = scan_chunk(current_state, chunk_lambdas)
            steps_run = chunk_size
        else:
            chunk_lambdas = lamb_array[num_chunks * chunk_size : ]
            current_state, _ = scan_remainder(current_state, chunk_lambdas)
            steps_run = remainder
            
        current_params = current_state[0]
        
        # 取该 chunk 的最后一个 lambda 值来计算当前的监控指标
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
        
        # 2. 完美刷新进度条，避免刷屏
        if pbar_GD is not None:
            pbar_GD.update(steps_run)
            pbar_GD.set_postfix({
                "Loss": f"{current_loss:.4f}", 
                "CCNR": f"{current_ccnr:.4f}", 
                "PPT": f"{current_ppt:.2e}",
                "Lamb": f"{current_lamb_val:.2f}" # 显示当前的 lambda 方便监控
            })

    if pbar_GD is not None:
        pbar_GD.close()

    # 最终结果重构
    final_params = current_state[0]
    params1 = jnp.matmul(jnp.conj(final_params.T), final_params) / jnp.trace(jnp.matmul(jnp.conj(final_params.T), final_params))
    
    return params1, ccnr_track, ppt_track, timel_GD, loss1

def plot_density_matrix_heatmap(rho, title="Searched Density Matrix"):
    # 转换为 numpy 格式，以防它是 JAX array 或 Qobj
    rho_np = np.array(rho) 
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 设定统一的颜色范围，让实部和虚部的颜色比例一致
    # 密度矩阵的元素绝对值通常不会超过 1
    vmax = max(np.max(np.abs(np.real(rho_np))), np.max(np.abs(np.imag(rho_np))))
    vmin = -vmax
    
    # 绘制实部 (cmap="RdBu_r" 是蓝-白-红的渐变色，0 是白色)
    sns.heatmap(np.real(rho_np), ax=axes[0], cmap="RdBu_r", 
                center=0, vmin=vmin, vmax=vmax, 
                annot=False, cbar=True, square=True)
    axes[0].set_title(f"{title} - Real Part")
    
    # 绘制虚部
    sns.heatmap(np.imag(rho_np), ax=axes[1], cmap="RdBu_r", 
                center=0, vmin=vmin, vmax=vmax, 
                annot=False, cbar=True, square=True)
    axes[1].set_title(f"{title} - Imaginary Part")
    
    plt.tight_layout()
    plt.show()
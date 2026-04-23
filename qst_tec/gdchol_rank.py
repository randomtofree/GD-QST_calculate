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
                       lr=2e-1, decay=0.01, lamb: float=0.01, 
                       tqdm_off=False, record_freq=100):
    """
    修改自 GD-QST，专用于搜索 Bound Entangled States (BES)。
    采用 jax.lax.scan 与 chunking 机制重构，大幅减少 Python 调度开销。
    输出接口与原版完全一致。
    """
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
    
    # 【优化 1】：将监控指标计算从核心更新循环中剥离
    @jax.jit
    def compute_metrics(current_params, current_lamb):
        # 仅在需要记录时，才重构 rho 并计算耗时的 CCNR 和 PPT
        t_dag_t = jnp.matmul(jnp.conj(current_params.T), current_params)
        rho = t_dag_t / jnp.trace(t_dag_t)
        
        loss_val = cost(current_params, current_lamb)
        ccnr_val = calculate_ccnr(rho)
        ppt_val = partial_transpose_b_penalty(rho)
        return loss_val, ccnr_val, ppt_val

    # 【优化 2】：lax.scan 的纯净单步更新函数（仅保留梯度和优化器状态计算）
    def scan_step(state, step_idx):
        current_params, current_opt_state = state
        
        # 计算梯度
        loss_val, grad_f = jax.value_and_grad(cost, argnums=0)(current_params, lamb)
        grads = jnp.conj(grad_f)
        
        # 更新参数
        updates, new_opt_state = gradient_transform.update(grads, current_opt_state, current_params)
        new_params = optax.apply_updates(current_params, updates)
        
        return (new_params, new_opt_state), None

    # 【优化 3】：利用 record_freq 作为 chunk_size
    chunk_size = record_freq
    num_chunks = iterations // chunk_size
    remainder = iterations % chunk_size

    @jax.jit
    def scan_chunk(current_state):
        # 编译 chunk_size 次循环，完全在硬件底层运行
        return jax.lax.scan(scan_step, current_state, jnp.arange(chunk_size))
        
    @jax.jit
    def scan_remainder(current_state):
        # 处理不能被 chunk_size 整除的剩余迭代
        return jax.lax.scan(scan_step, current_state, jnp.arange(remainder))

    tot_time = 0
    current_state = (params, opt_state)
    
    # 进度条现在显示的是 Chunk 的进度，而不是单步的进度
    total_steps = num_chunks + (1 if remainder > 0 else 0)
    pbar_GD = range(total_steps) if tqdm_off else tqdm(range(total_steps), desc="BES Search")

    for i in pbar_GD:
        start = time.time()
        
        # 1. 批量在底层执行参数更新
        if i < num_chunks:
            current_state, _ = scan_chunk(current_state)
        else:
            current_state, _ = scan_remainder(current_state)
            
        # 2. 提取当前 Chunk 结束后的最新参数
        current_params = current_state[0]
        
        # 3. 计算本阶段的监控指标
        loss_val, ccnr_val, ppt_val = compute_metrics(current_params, lamb)
        
        # JAX 是异步的，调用 float() 会隐式触发 block_until_ready() 同步计算
        current_loss = float(loss_val)
        current_ccnr = float(ccnr_val)
        current_ppt = float(ppt_val)
        
        end = time.time()
        tot_time += (end - start)
        
        # 4. 记录数据（输出格式和频率与原版完全相同）
        loss1.append(current_loss)
        ccnr_track.append(current_ccnr)
        ppt_track.append(current_ppt)
        timel_GD.append(tot_time)
        
        if not tqdm_off:
            pbar_GD.set_description(f"Loss: {current_loss:.4f} | CCNR: {current_ccnr:.4f} | PPT: {current_ppt:.2e}")

    # 最终结果重构，返回结构保持原样
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
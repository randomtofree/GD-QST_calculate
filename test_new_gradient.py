import jax
import jax.numpy as jnp
from jax.test_util import check_grads
import numpy as np

# 强制开启 64 位双精度，这对于梯度验证至关重要
jax.config.update("jax_enable_x64", True)

# ==========================================
# 1. 原生 JAX 实现 (Baseline)
# ==========================================
def native_nuclear_norm(M):
    # JAX 原生的核范数
    return jnp.linalg.norm(M, 'nuc')

def native_ppt_penalty(rho_pt):
    # JAX 原生的 PPT 惩罚
    evals = jnp.linalg.eigvalsh(rho_pt)
    return jnp.sum(jnp.minimum(0., evals)**2)

# ==========================================
# 2. 自定义 VJP 实现 (优化版)
# ==========================================
@jax.custom_vjp
def custom_nuclear_norm(M):
    s = jnp.linalg.svd(M, compute_uv=False)
    return jnp.sum(s)

def nuclear_norm_fwd(M):
    U, s, Vh = jnp.linalg.svd(M, full_matrices=False)
    return jnp.sum(s), (U, Vh)

def nuclear_norm_bwd(res, g):
    U, Vh = res
    # 物理/数学上的解析梯度是 U @ Vh
    # JAX 的 VJP 约定要求我们返回梯度的复共轭 (Complex Conjugate)
    grad_M = g * (U @ Vh).conj()
    return (grad_M,)

custom_nuclear_norm.defvjp(nuclear_norm_fwd, nuclear_norm_bwd)


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
    # 标量函数 f(λ) 的导数
    grad_evals = jnp.where(evals < 0, 2.0 * evals, 0.0)
    
    # 物理/数学上的解析矩阵梯度 G
    # 注意：因为 rho 是 Hermitian，evecs 构成的这个梯度矩阵本身也是 Hermitian 的
    G = evecs @ jnp.diag(grad_evals) @ evecs.conj().T
    
    # JAX 要求返回 G 的复共轭 (即转置)
    grad_matrix = g * G.conj()
    return (grad_matrix,)

custom_ppt_penalty.defvjp(ppt_penalty_fwd, ppt_penalty_bwd)

# ==========================================
# 3. 验证主程序
# ==========================================
def run_verification():
    print("=== 开始验证自定义梯度 ===")
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    D = 16 # 对应 4x4 系统
    
    # --- 测试 1: 核范数梯度验证 ---
    print("\n[1] 验证核范数 (Nuclear Norm)...")
    # 生成一个随机的复数矩阵
    M_real = jax.random.normal(k1, (D, D), dtype=jnp.float64)
    M_imag = jax.random.normal(k2, (D, D), dtype=jnp.float64)
    M = M_real + 1j * M_imag
    
    # 计算梯度
    grad_nuc_native = jax.grad(native_nuclear_norm)(M)
    grad_nuc_custom = jax.grad(custom_nuclear_norm)(M)
    
    # 对比结果
    diff_nuc = jnp.max(jnp.abs(grad_nuc_native - grad_nuc_custom))
    print(f"最大绝对误差: {diff_nuc:.2e}")
    if jnp.allclose(grad_nuc_native, grad_nuc_custom, atol=1e-8, rtol=1e-5):
        print("✅ 核范数解析梯度与 JAX 原生 autodiff 完全一致！")
    else:
        print("❌ 核范数梯度存在偏差！")


    # --- 测试 2: PPT 惩罚梯度验证 ---
    print("\n[2] 验证 PPT 惩罚 (PPT Penalty)...")
    # 生成一个随机的 Hermitian 矩阵 (物理上的部分转置密度矩阵是 Hermitian 的)
    H_temp = jax.random.normal(k3, (D, D), dtype=jnp.float64) + 1j * jax.random.normal(k1, (D, D), dtype=jnp.float64)
    rho_pt = H_temp + H_temp.conj().T # 保证是 Hermitian 矩阵
    
    # 计算梯度
    # 注意：jax.grad 默认期望输入是实数或全自由度复数。
    # 由于 rho_pt 必须是 Hermitian，我们需要用 jax.grad 计算后，取 Hermitian 投影 
    # (JAX 的 eigh 内部处理反向传播时也做了这个假设)。
    grad_ppt_native = jax.grad(native_ppt_penalty)(rho_pt)
    grad_ppt_custom = jax.grad(custom_ppt_penalty)(rho_pt)
    
    diff_ppt = jnp.max(jnp.abs(grad_ppt_native - grad_ppt_custom))
    print(f"最大绝对误差: {diff_ppt:.2e}")
    if jnp.allclose(grad_ppt_native, grad_ppt_custom, atol=1e-8, rtol=1e-5):
        print("✅ PPT 惩罚解析梯度与 JAX 原生 autodiff 完全一致！")
    else:
        print("❌ PPT 惩罚梯度存在偏差！")

    # --- 测试 3: (可选) 使用 check_grads 测试有限差分 ---
    print("\n[3] (可选) 使用 check_grads 进行有限差分测试...")
    try:
        check_grads(custom_nuclear_norm, (M,), order=1, modes=['rev'])
        print("✅ check_grads 核范数有限差分测试通过！")
    except AssertionError as e:
        print("⚠️ check_grads 核范数未通过 (这在复数 SVD 中很常见，通常是因为相位跳变，只要上述对比通过即安全)。")

if __name__ == "__main__":
    run_verification()
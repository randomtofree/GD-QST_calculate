import jax
import jax.numpy as jnp
import optax
import time

# 导入你 gdchol_rank.py 中已经写好的自定义梯度 PPT 函数
# 注意：一定要用上一轮对话中加上了 .conj() 修复了 Bug 的版本！
from qst_tec.gdchol_rank import custom_ppt_penalty, custom_nuclear_norm

# ==========================================
# 第一部分：构造可分态 (验证器的 Ansatz)
# ==========================================
def build_separable_state(A, B):
    """
    基于 Carathéodory 思想的纯直积态凸组合。
    A, B 形状: (K, d)
    """
    def single_product_state(a, b):
        ab = jnp.kron(a, b) # 张量积
        return jnp.outer(ab, jnp.conj(ab)) # 构造密度矩阵

    # vmap 并行计算 K 个态，并在 K 维度上求和
    unnormalized_rho = jnp.sum(jax.vmap(single_product_state)(A, B), axis=0)
    # 归一化，保证迹为 1
    return unnormalized_rho / jnp.trace(unnormalized_rho)

def distance_sq(rho1, rho2):
    """计算两个矩阵间的 Frobenius 范数平方 (几何距离)"""
    return jnp.sum(jnp.abs(rho1 - rho2)**2).real

# ==========================================
# 第二部分：定义双方的 Loss 函数
# ==========================================

# 1. 验证器 Loss：只求最小化几何距离
def verifier_loss(params_V, rho_target):
    A, B = params_V
    rho_sep = build_separable_state(A, B)
    return distance_sq(rho_target, rho_sep)

# 2. 生成器 Loss：保持 PPT 的同时，最大化几何距离
def generator_loss(T, params_V, lamb):
    # 构造当前目标态
    t_dag_t = jnp.conj(T.T) @ T
    rho_target = t_dag_t / jnp.trace(t_dag_t)
    
    # 构造当前最优可分态 (此时验证器参数固定)
    A, B = params_V
    rho_sep = build_separable_state(A, B)
    
    # PPT 惩罚
    # 再次强调：一定要用修复了自定义 VJP 的版本！
    d = int(rho_target.shape[0] ** 0.5)
    rho_tensor = rho_target.reshape((d, d, d, d))
    rho_pt_tensor = jnp.transpose(rho_tensor, (0, 3, 2, 1)) # 正确的部分转置
    rho_pt = rho_pt_tensor.reshape((d**2, d**2))
    rho_pt = (rho_pt + rho_pt.conj().T) / 2.0
    ppt_pen = custom_ppt_penalty(rho_pt)
    
    dist = distance_sq(rho_target, rho_sep)
    
    # Min-Max 核心：生成器希望 loss 越小越好。
    # 所以它需要极小的 PPT 惩罚，和极大的 dist (所以是减去 dist)
    return lamb * ppt_pen - dist, (ppt_pen, dist)

# ==========================================
# 第三部分：JIT 编译双方的更新步 (交替更新)
# ==========================================

# 初始化两个优化器
# 通常在 GAN 中，验证器（判别器）学习率要快一些，保证它总是能追上生成器
lr_G = 0.01
lr_V = 0.05
opt_G = optax.adam(lr_G)
opt_V = optax.adam(lr_V)

@jax.jit
def update_verifier(params_V, opt_state_V, rho_target):
    loss_val, grads = jax.value_and_grad(verifier_loss)(params_V, rho_target)
    # JAX 复数梯度坑：对 Tuple 里的每一个矩阵取共轭
    grads = jax.tree.map(lambda g: jnp.conj(g), grads)
    updates, new_opt_state = opt_V.update(grads, opt_state_V, params_V)
    new_params = optax.apply_updates(params_V, updates)
    return new_params, new_opt_state, loss_val

@jax.jit
def update_generator(T, opt_state_G, params_V, lamb):
    (loss_val, (ppt, dist)), grads = jax.value_and_grad(generator_loss, has_aux=True)(T, params_V, lamb)
    grads = jnp.conj(grads) # JAX 复数梯度共轭
    updates, new_opt_state = opt_G.update(grads, opt_state_G, T)
    new_T = optax.apply_updates(T, updates)
    return new_T, new_opt_state, loss_val, ppt, dist

# ==========================================
# 第四部分：开始对抗训练 (The Game)
# ==========================================

def train_quantum_gan(d=4, K=64, rank_T=8, steps=5000, lamb=100.0):
    key = jax.random.PRNGKey(42)
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    
    # 随机初始化双方参数
    T = jax.random.normal(k1, (rank_T, d**2)) + 1j * jax.random.normal(k2, (rank_T, d**2))
    A = jax.random.normal(k3, (K, d)) + 1j * jax.random.normal(k4, (K, d))
    B = jax.random.normal(k4, (K, d)) + 1j * jax.random.normal(k3, (K, d))
    params_V = (A, B)
    
    opt_state_G = opt_G.init(T)
    opt_state_V = opt_V.init(params_V)
    
    print(f"开始对抗搜索: d={d}, K={K}, 目标态秩={rank_T}")
    start_time = time.time()
    
    for i in range(steps):
        # --- 计算当前目标态 (作为验证器的靶子) ---
        t_dag_t = jnp.conj(T.T) @ T
        rho_target = t_dag_t / jnp.trace(t_dag_t)
        
        # --- 步骤 A：训练验证器 (让它充分拟合) ---
        # GAN 的经典策略：判别器多跑几步，保证评估准确
        for _ in range(5): 
            params_V, opt_state_V, v_loss = update_verifier(params_V, opt_state_V, rho_target)
            
        # --- 步骤 B：训练生成器 (让它逃脱拟合) ---
        T, opt_state_G, g_loss, ppt, dist = update_generator(T, opt_state_G, params_V, lamb)
        
        if i % 500 == 0:
            print(f"Step {i:4d} | 生成器逃逸距离: {dist:.5f} | 验证器追赶误差: {v_loss:.5f} | PPT惩罚: {ppt:.2e}")
            
    print(f"耗时: {time.time() - start_time:.2f}s")
    return T, params_V

# 运行主程序
if __name__ == '__main__':
    # 配置使用双精度
    jax.config.update("jax_enable_x64", True)
    # 对于 4x4，先尝试 K=64 (不用一开始就 256)。
    final_T, final_V = train_quantum_gan(d=4, K=256, rank_T=8, steps=5000, lamb=50.0)

    # 在训练结束后提取最终的态
    t_dag_t = jnp.conj(final_T.T) @ final_T
    rho_final = t_dag_t / jnp.trace(t_dag_t)

    # 打印它的 CCNR 看看！
    print("最终态的 CCNR 值:", custom_nuclear_norm(rho_final))
import numpy as np
from qutip import Qobj, qeye
from itertools import product
from qutip import tensor

def get_generalized_gell_mann(d):
    """
    生成任意维度 d 的广义盖尔曼矩阵基底。
    返回一个长度为 d^2 的列表，包含 QuTiP 的 Qobj 对象。
    列表的第 0 个元素为 d 维单位矩阵 I_d。
    """
    # 1. 放入单位矩阵
    matrices = [qeye(d)]
    
    # 2. 生成非对角矩阵 (对称和反对称)
    for j in range(d):
        for k in range(j + 1, d):
            # 构造对称矩阵 (Symmetric)
            sym = np.zeros((d, d), dtype=np.complex128)
            sym[j, k] = 1.0
            sym[k, j] = 1.0
            matrices.append(Qobj(sym))
            
            # 构造反对称矩阵 (Antisymmetric)
            anti = np.zeros((d, d), dtype=np.complex128)
            anti[j, k] = -1.0j
            anti[k, j] =  1.0j
            matrices.append(Qobj(anti))
            
    # 3. 生成对角矩阵 (Diagonal)
    for l in range(1, d):
        diag = np.zeros((d, d), dtype=np.complex128)
        # 前 l 个对角元设为 1
        for j in range(l):
            diag[j, j] = 1.0
        # 第 l+1 个对角元设为 -l (索引为 l)
        diag[l, l] = -l
        
        # 归一化系数，确保 Tr(Lambda^2) = 2
        prefactor = np.sqrt(2.0 / (l * (l + 1)))
        matrices.append(Qobj(prefactor * diag))
        
    return matrices
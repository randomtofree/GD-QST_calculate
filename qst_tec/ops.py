import tensorflow as tf

import numpy as np

from qutip import coherent_dm, Qobj, coherent


def expect(ops, rhos):
    """
    Calculates expectation values for a batch of density matrices
    for list of operators.
    
    Args:
        ops (`tf.Tensor`): a 3D tensor (N, hilbert_size, hilbert_size) of N
                                         measurement operators
        rhos (`tf.Tensor`): a 4D tensor (batch_size, hilbert_size, hilbert_size)
                            representing N density matrices        

    Returns:
        expectations (:class:`tensorflow.Tensor`): A 4D tensor (batch_size, N, 1)
                                                   giving expectation values for the
                                                   (N, N) grid of operators for
                                                   all the density matrices (batch_size).
    """
    products = tf.einsum("aij, bjk->baik", ops, rhos)
    traces = tf.linalg.trace(products)
    expectations = tf.math.real(traces)
    return expectations


def random_alpha(radius, inner_radius=0):
    """
    Generates a random complex values within a circle
    
    Args:
        radius (float): Radius for the values
        inner_radius (float): Inner radius which defaults to 0.
    """
    radius = np.random.uniform(inner_radius, radius)
    phi = np.random.uniform(-np.pi, np.pi)
    return radius * np.exp(1j * phi)


def dm_to_tf(rhos):
    """
    Convert a list of qutip density matrices to TensorFlow 
    density matrices

    Args:
        rhos (list of `qutip.Qobj`): List of N qutip density matrices

    Returns:
        tf_dms (:class:`tensorflow.Tensor`): A 3D tensor (N, hilbert_size, hilbert_size) of N
                                         density matrices

    """
    tf_dms = tf.convert_to_tensor(
        [tf.complex(rho.full().real, rho.full().imag) for rho in rhos]
    )
    return tf_dms


def husimi_ops(hilbert_size, betas):
    """
    Constructs a list of TensorFlow operators for the Husimi Q function
    measurement at beta values.
    
    Args:
        hilbert_size (int): The hilbert size dimension for the operators
        betas (list/array): N complex values to construct the operator
        
    Returns:
        ops (:class:`tensorflow.Tensor`): A 3D tensor (N, hilbert_size, hilbert_size) of N
                                         operators
    """
    basis = []
    for beta in betas:
        basis.append(coherent_dm(hilbert_size, beta))

    return dm_to_tf(basis)


def tf_to_dm(rhos):
    """
    Convert a tensorflow density matrix to qutip density matrix

    Args:
        rhos (`tf.Tensor`): a 4D tensor (N, hilbert_size, hilbert_size)
                            representing density matrices        
        
    Returns:
        rho_gen (list of :class:`qutip.Qobj`): A list of N density matrices

    """
    rho_gen = [Qobj(rho.numpy()) for rho in rhos]
    return rho_gen


def clean_cholesky(img):
    """
    Cleans an input matrix to make it the Cholesky decomposition matrix T
    
    Args:
        img (`tf.Tensor`): a 4D tensor (batch_size, hilbert_size, hilbert_size, 2)
                           representing batch_size random outputs from a neural netowrk.
                           The last dimension is for separating the real and imaginary part

    Returns:
        T (`tf.Tensor`): a 3D tensor (N, hilbert_size, hilbert_size)
                           representing N T matrices
    """
    real = img[:, :, :, 0]
    imag = img[:, :, :, 1]

    diag_all = tf.linalg.diag_part(imag, k=0, padding_value=0)
    diags = tf.linalg.diag(diag_all)

    imag = imag - diags
    imag = tf.linalg.band_part(imag, -1, 0)
    real = tf.linalg.band_part(real, -1, 0)
    T = tf.complex(real, imag)
    return T


def density_matrix_from_T(tmatrix):
    """
    Gets density matrices from T matrices and normalizes them.
    
    Args:
        tmatrix (`tf.Tensor`): 3D tensor (N, hilbert_size, hilbert_size)
                           representing N valid T matrices

    Returns:
        rho (`tf.Tensor`): 3D tensor (N, hilbert_size, hilbert_size)
                           representing N density matrices
    """
    T = tmatrix
    T_dagger = tf.transpose(T, perm=[0, 2, 1], conjugate=True)
    proper_dm = tf.matmul(T, T_dagger)
    all_traces = tf.linalg.trace(proper_dm)
    all_traces = tf.reshape(1 / all_traces, (-1, 1))
    rho = tf.einsum("bij,bk->bij", proper_dm, all_traces)

    return rho


def convert_to_real_ops(ops):
    """
    Converts a batch of TensorFlow operators to something that a neural network
    can take as input.
    
    Args:
        ops (`tf.Tensor`): a 4D tensor (batch_size, N, hilbert_size, hilbert_size) of N
                           measurement operators

    Returns:
        tf_ops (`tf.Tensor`): a 4D tensor (batch_size, hilbert_size, hilbert_size, 2*N) of N
                           measurement operators converted into real matrices
    """
    tf_ops = tf.transpose(ops, perm=[0, 2, 3, 1])
    tf_ops = tf.concat([tf.math.real(tf_ops), tf.math.imag(tf_ops)], axis=-1)
    return tf_ops


def convert_to_complex_ops(ops):
    """
    Converts a batch of TensorFlow operators to something that a neural network
    can take as input.
    
    Args:
        ops (`tf.Tensor`): a 4D tensor (batch_size, N, hilbert_size, hilbert_size) of N
                           measurement operators

    Returns:
        tf_ops (`tf.Tensor`): a 4D tensor (batch_size, hilbert_size, hilbert_size, 2*N) of N
                           measurement operators converted into real matrices
    """
    shape = ops.shape
    num_points = shape[-1]
    tf_ops = tf.complex(ops[..., :int(num_points/2)], ops[..., int(num_points/2):])
    tf_ops = tf.transpose(tf_ops, perm=[0, 3, 1, 2])
    return tf_ops


def batched_expect(ops, rhos):
    """
    Calculates expectation values for a batch of density matrices
    for a batch of N sets of operators
    
    Args:
        ops (`tf.Tensor`): a 4D tensor (batch_size, N, hilbert_size, hilbert_size) of N
                                         measurement operators
        rhos (`tf.Tensor`): a 4D tensor (batch_size, hilbert_size, hilbert_size)

    Returns:
        expectations (:class:`tensorflow.Tensor`): A 4D tensor (batch_size, N)
                                                   giving expectation values for the
                                                   N grid of operators for
                                                   all the density matrices (batch_size).
    """
    products = tf.einsum("bnij, bjk->bnik", ops, rhos)
    traces = tf.linalg.trace(products)
    expectations = tf.math.real(traces)
    return expectations


def tf_fidelity(A, B):
    """
    Calculated the fidelity between batches of tensors A and B
    """
    sqrtmA = tf.matrix_square_root(A)
    temp = tf.matmul(sqrtmA, B)
    temp2 = tf.matmul(temp, sqrtmA)
    fidel = tf.linalg.trace(tf.linalg.sqrtm(temp2))**2
    return tf.math.real(fidel)


def cat(N, alpha, S=None, mu=None):
    """
    Generates a cat state. For a detailed discussion on the definition
    see `Albert, Victor V. et al. “Performance and Structure of Single-Mode Bosonic Codes.” Physical Review A 97.3 (2018) <https://arxiv.org/abs/1708.05010>`_
    and `Ahmed, Shahnawaz et al., “Classification and reconstruction of quantum states with neural networks.” Journal <https://arxiv.org/abs/1708.05010>`_
    
    Args:
    -----
        N (int): Hilbert size dimension.
        alpha (complex64): Complex number determining the amplitude.
        S (int): An integer >= 0 determining the number of coherent states used
                 to generate the cat superposition. S = {0, 1, 2, ...}.
                 corresponds to {2, 4, 6, ...} coherent state superpositions.
        mu (int): An integer 0/1 which generates the logical 0/1 encoding of 
                  a computational state for the cat state.


    Returns:
    -------
        cat (:class:`qutip.Qobj`): Cat state density matrix
    """
    if S == None:
        S = 0

    if mu is None:
        mu = 0

    kend = 2 * S + 1
    cstates = 0 * (coherent(N, 0))

    for k in range(0, int((kend + 1) / 2)):
        sign = 1

        if k >= S:
            sign = (-1) ** int(mu > 0.5)

        prefactor = np.exp(1j * (np.pi / (S + 1)) * k)

        cstates += sign * coherent(N, prefactor * alpha * (-((1j) ** mu)))
        cstates += sign * coherent(N, -prefactor * alpha * (-((1j) ** mu)))

    rho = cstates * cstates.dag()
    return rho.unit()
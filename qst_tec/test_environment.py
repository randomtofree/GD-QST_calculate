import subprocess
import os


def check_cuda_simple():
    print("=== CUDA 环境检查 ===")

    # 检查CUDA版本
    try:
        nvcc_output = subprocess.check_output(['nvcc', '--version']).decode()
        print(f"NVCC版本信息:\n{nvcc_output}")
    except:
        print("未找到NVCC,请检查CUDA安装")

    # 检查环境变量
    cuda_path = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    print(f"CUDA路径: {cuda_path}")

    # 检查nvidia-smi
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi']).decode()
        print(f"\nGPU信息:\n{nvidia_smi}")
    except:
        print("未找到nvidia-smi,请检查GPU驱动")


if __name__ == "__main__":
    check_cuda_simple()
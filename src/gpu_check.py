import os
import cudf, cuml, dask_cuda, rmm

def check_gpu():
    print("CUDA:", os.popen("nvcc --version").read().splitlines()[-1])
    print(f"cudf: {cudf.__version__}")
    print(f"cuml: {cuml.__version__}")
    print(f"dask_cuda: {dask_cuda.__version__}")
    print(f"rmm: {rmm.__version__}")
print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
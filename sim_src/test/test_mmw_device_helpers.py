import numpy as np

from sim_src.alg.mmw import mmw


def test_mmw_tensor_conversion_returns_cpu_tensor_when_gpu_disabled():
    alg = mmw(nit=1, device="cpu")
    arr = np.eye(3, dtype=float)
    tensor = alg._to_torch(arr, device="cpu")
    assert tuple(tensor.shape) == (3, 3)
    assert str(tensor.device) == "cpu"


def test_mmw_selects_requested_cuda_device_when_available():
    import sim_src.alg.mmw as mmw_module

    if mmw_module.torch is None:
        return

    original = mmw_module.torch.cuda.is_available
    try:
        mmw_module.torch.cuda.is_available = lambda: True
        alg = mmw(nit=1, device=mmw_module.torch.device("cuda:0"))
        assert str(alg.device) == "cuda:0"
    finally:
        mmw_module.torch.cuda.is_available = original

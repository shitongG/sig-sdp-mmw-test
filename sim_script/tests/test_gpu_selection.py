from sim_src.util import resolve_torch_device


def test_resolve_torch_device_defaults_to_cpu_when_disabled():
    device = resolve_torch_device(use_gpu=False, gpu_id=0)
    assert str(device) == "cpu"


def test_resolve_torch_device_uses_selected_gpu_id_when_available():
    import sim_src.util as util

    if util.torch is None:
        return

    original = util.torch.cuda.is_available
    try:
        util.torch.cuda.is_available = lambda: True
        device = resolve_torch_device(use_gpu=True, gpu_id=2)
        assert str(device) == "cuda:2"
    finally:
        util.torch.cuda.is_available = original

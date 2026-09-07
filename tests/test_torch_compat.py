import torch

from videoclean.adapters.torch_compat import ensure_torch_compiler_compat


def test_ensure_torch_compiler_compat_provides_is_compiling():
    # Simulate torch 2.2: module exists, attribute may be missing.
    compiler = torch.compiler
    had = hasattr(compiler, "is_compiling")
    if had:
        original = compiler.is_compiling
        delattr(compiler, "is_compiling")
    try:
        import videoclean.adapters.torch_compat as compat

        compat._patched = False
        ensure_torch_compiler_compat()
        assert hasattr(torch.compiler, "is_compiling")
        assert torch.compiler.is_compiling() is False
    finally:
        if had:
            compiler.is_compiling = original
        else:
            # leave the shim in place for the rest of the suite on torch 2.2
            pass

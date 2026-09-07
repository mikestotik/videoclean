from __future__ import annotations

_patched = False


def ensure_torch_compiler_compat() -> None:
    """transformers>=4.56 fast image ops call torch.compiler.is_compiling (torch>=2.3).

    On Intel Mac the newest official torch wheel is 2.2.2, which has torch.compiler
    but not is_compiling. Provide a no-op so Sam2/processor code can run.
    """
    global _patched
    if _patched:
        return
    import torch

    compiler = getattr(torch, "compiler", None)
    if compiler is not None and not hasattr(compiler, "is_compiling"):
        compiler.is_compiling = lambda: False  # type: ignore[attr-defined]
    _patched = True

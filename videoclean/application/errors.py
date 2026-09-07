from __future__ import annotations


class PipelineError(RuntimeError):
    """Something the user can fix: bad flag, missing binary, missing model."""


class AdapterNotImplemented(PipelineError):
    """Port exists, this adapter is not wired yet."""


class AdapterUnavailable(PipelineError):
    """Adapter is wired but cannot run on this machine (no model, no GPU, …)."""


class DeviceUnavailable(PipelineError):
    """--device requested a backend torch cannot use here."""


class JobCancelled(PipelineError):
    """Cleanup job was cancelled before or during a run."""


class DownloadCancelled(PipelineError):
    """Model download was cancelled before it finished."""

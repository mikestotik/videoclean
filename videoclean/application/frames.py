from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from videoclean.application.errors import PipelineError


class LazyFrames:
    """Random-access decoded frames. Keeps at most cache_size images in RAM."""

    def __init__(self, paths: list[Path], read, cache_size: int = 24) -> None:
        self.paths = list(paths)
        self._read = read
        self._cache_size = max(1, cache_size)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        i = int(idx)
        if i < 0:
            i += len(self)
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        path = self.paths[i]
        image = self._read(path)
        if image is None:
            raise PipelineError(f"cannot read {path}")
        self._cache[i] = image
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return image

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
